from __future__ import annotations
import contextlib
import datetime
from functools import partial
from functools import wraps
import json
import logging
from numbers import Number
import threading
import time
from typing import Any
from typing import Callable
from typing import cast
from typing import Mapping
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import TYPE_CHECKING
from typing import Union
from decorator import decorate
from . import exception
from .api import BackendArguments
from .api import BackendFormatted
from .api import CachedValue
from .api import CacheMutex
from .api import CacheReturnType
from .api import CantDeserializeException
from .api import KeyType
from .api import MetaDataType
from .api import NO_VALUE
from .api import SerializedReturnType
from .api import Serializer
from .api import ValuePayload
from .backends import _backend_loader
from .backends import register_backend  # noqa
from .proxy import ProxyBackend
from .util import function_key_generator
from .util import function_multi_key_generator
from .util import repr_obj
from .. import Lock
from .. import NeedRegenerationException
from ..util import coerce_string_conf
from ..util import memoized_property
from ..util import NameRegistry
from ..util import PluginLoader
from ..util.typing import Self
def get_or_create_multi(self, keys: Sequence[KeyType], creator: Callable[[], ValuePayload], expiration_time: Optional[float]=None, should_cache_fn: Optional[Callable[[ValuePayload], bool]]=None) -> Sequence[ValuePayload]:
    """Return a sequence of cached values based on a sequence of keys.

        The behavior for generation of values based on keys corresponds
        to that of :meth:`.Region.get_or_create`, with the exception that
        the ``creator()`` function may be asked to generate any subset of
        the given keys.   The list of keys to be generated is passed to
        ``creator()``, and ``creator()`` should return the generated values
        as a sequence corresponding to the order of the keys.

        The method uses the same approach as :meth:`.Region.get_multi`
        and :meth:`.Region.set_multi` to get and set values from the
        backend.

        If you are using a :class:`.CacheBackend` or :class:`.ProxyBackend`
        that modifies values, take note this function invokes
        ``.set_multi()`` for newly generated values using the same values it
        returns to the calling function. A correct implementation of
        ``.set_multi()`` will not modify values in-place on the submitted
        ``mapping`` dict.

        :param keys: Sequence of keys to be retrieved.

        :param creator: function which accepts a sequence of keys and
         returns a sequence of new values.

        :param expiration_time: optional expiration time which will override
         the expiration time already configured on this :class:`.CacheRegion`
         if not None.   To set no expiration, use the value -1.

        :param should_cache_fn: optional callable function which will receive
         each value returned by the "creator", and will then return True or
         False, indicating if the value should actually be cached or not.  If
         it returns False, the value is still returned, but isn't cached.

        .. versionadded:: 0.5.0

        .. seealso::


            :meth:`.CacheRegion.cache_multi_on_arguments`

            :meth:`.CacheRegion.get_or_create`

        """

    def get_value(key):
        value = values.get(key, NO_VALUE)
        if self._is_cache_miss(value, orig_key):
            return (value.payload, 0)
        else:
            ct = cast(CachedValue, value).metadata['ct']
            if self.region_invalidator.is_soft_invalidated(ct):
                if expiration_time is None:
                    raise exception.DogpileCacheException('Non-None expiration time required for soft invalidation')
                ct = time.time() - expiration_time - 0.0001
            return (value.payload, ct)

    def gen_value() -> ValuePayload:
        raise NotImplementedError()

    def async_creator(mutexes, key, mutex):
        mutexes[key] = mutex
    if expiration_time is None:
        expiration_time = self.expiration_time
    if expiration_time == -1:
        expiration_time = None
    sorted_unique_keys = sorted(set(keys))
    if self.key_mangler:
        mangled_keys = [self.key_mangler(k) for k in sorted_unique_keys]
    else:
        mangled_keys = sorted_unique_keys
    orig_to_mangled = dict(zip(sorted_unique_keys, mangled_keys))
    values = dict(zip(mangled_keys, self._get_multi_from_backend(mangled_keys)))
    mutexes: Mapping[KeyType, Any] = {}
    for orig_key, mangled_key in orig_to_mangled.items():
        with Lock(self._mutex(mangled_key), gen_value, lambda: get_value(mangled_key), expiration_time, async_creator=lambda mutex: async_creator(mutexes, orig_key, mutex)):
            pass
    try:
        if mutexes:
            keys_to_get = sorted(mutexes)
            with self._log_time(keys_to_get):
                new_values = creator(*keys_to_get)
            values_w_created = {orig_to_mangled[k]: self._value(v) for k, v in zip(keys_to_get, new_values)}
            if expiration_time is None and self.region_invalidator.was_soft_invalidated():
                raise exception.DogpileCacheException('Non-None expiration time required for soft invalidation')
            if not should_cache_fn:
                self._set_multi_cached_value_to_backend(values_w_created)
            else:
                self._set_multi_cached_value_to_backend({k: v for k, v in values_w_created.items() if should_cache_fn(v.payload)})
            values.update(values_w_created)
        return [values[orig_to_mangled[k]].payload for k in keys]
    finally:
        for mutex in mutexes.values():
            mutex.release()