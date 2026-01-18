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
def cache_multi_on_arguments(self, namespace: Optional[str]=None, expiration_time: Union[float, ExpirationTimeCallable, None]=None, should_cache_fn: Optional[Callable[[ValuePayload], bool]]=None, asdict: bool=False, to_str: ToStr=str, function_multi_key_generator: Optional[FunctionMultiKeyGenerator]=None) -> Callable[[Callable[..., Sequence[ValuePayload]]], Callable[..., Union[Sequence[ValuePayload], Mapping[KeyType, ValuePayload]]]]:
    """A function decorator that will cache multiple return
        values from the function using a sequence of keys derived from the
        function itself and the arguments passed to it.

        This method is the "multiple key" analogue to the
        :meth:`.CacheRegion.cache_on_arguments` method.

        Example::

            @someregion.cache_multi_on_arguments()
            def generate_something(*keys):
                return [
                    somedatabase.query(key)
                    for key in keys
                ]

        The decorated function can be called normally.  The decorator
        will produce a list of cache keys using a mechanism similar to
        that of :meth:`.CacheRegion.cache_on_arguments`, combining the
        name of the function with the optional namespace and with the
        string form of each key.  It will then consult the cache using
        the same mechanism as that of :meth:`.CacheRegion.get_multi`
        to retrieve all current values; the originally passed keys
        corresponding to those values which aren't generated or need
        regeneration will be assembled into a new argument list, and
        the decorated function is then called with that subset of
        arguments.

        The returned result is a list::

            result = generate_something("key1", "key2", "key3")

        The decorator internally makes use of the
        :meth:`.CacheRegion.get_or_create_multi` method to access the
        cache and conditionally call the function.  See that
        method for additional behavioral details.

        Unlike the :meth:`.CacheRegion.cache_on_arguments` method,
        :meth:`.CacheRegion.cache_multi_on_arguments` works only with
        a single function signature, one which takes a simple list of
        keys as arguments.

        Like :meth:`.CacheRegion.cache_on_arguments`, the decorated function
        is also provided with a ``set()`` method, which here accepts a
        mapping of keys and values to set in the cache::

            generate_something.set({"k1": "value1",
                                    "k2": "value2", "k3": "value3"})

        ...an ``invalidate()`` method, which has the effect of deleting
        the given sequence of keys using the same mechanism as that of
        :meth:`.CacheRegion.delete_multi`::

            generate_something.invalidate("k1", "k2", "k3")

        ...a ``refresh()`` method, which will call the creation
        function, cache the new values, and return them::

            values = generate_something.refresh("k1", "k2", "k3")

        ...and a ``get()`` method, which will return values
        based on the given arguments::

            values = generate_something.get("k1", "k2", "k3")

        .. versionadded:: 0.5.3 Added ``get()`` method to decorated
           function.

        Parameters passed to :meth:`.CacheRegion.cache_multi_on_arguments`
        have the same meaning as those passed to
        :meth:`.CacheRegion.cache_on_arguments`.

        :param namespace: optional string argument which will be
         established as part of each cache key.

        :param expiration_time: if not None, will override the normal
         expiration time.  May be passed as an integer or a
         callable.

        :param should_cache_fn: passed to
         :meth:`.CacheRegion.get_or_create_multi`. This function is given a
         value as returned by the creator, and only if it returns True will
         that value be placed in the cache.

        :param asdict: if ``True``, the decorated function should return
         its result as a dictionary of keys->values, and the final result
         of calling the decorated function will also be a dictionary.
         If left at its default value of ``False``, the decorated function
         should return its result as a list of values, and the final
         result of calling the decorated function will also be a list.

         When ``asdict==True`` if the dictionary returned by the decorated
         function is missing keys, those keys will not be cached.

        :param to_str: callable, will be called on each function argument
         in order to convert to a string.  Defaults to ``str()``.  If the
         function accepts non-ascii unicode arguments on Python 2.x, the
         ``unicode()`` builtin can be substituted, but note this will
         produce unicode cache keys which may require key mangling before
         reaching the cache.

        .. versionadded:: 0.5.0

        :param function_multi_key_generator: a function that will produce a
         list of keys. This function will supersede the one configured on the
         :class:`.CacheRegion` itself.

         .. versionadded:: 0.5.5

        .. seealso::

            :meth:`.CacheRegion.cache_on_arguments`

            :meth:`.CacheRegion.get_or_create_multi`

        """
    expiration_time_is_callable = callable(expiration_time)
    if function_multi_key_generator is None:
        _function_multi_key_generator = self.function_multi_key_generator
    else:
        _function_multi_key_generator = function_multi_key_generator

    def get_or_create_for_user_func(key_generator: Callable[..., Sequence[KeyType]], user_func: Callable[..., Sequence[ValuePayload]], *arg: Any, **kw: Any) -> Union[Sequence[ValuePayload], Mapping[KeyType, ValuePayload]]:
        cache_keys = arg
        keys = key_generator(*arg, **kw)
        key_lookup = dict(zip(keys, cache_keys))

        @wraps(user_func)
        def creator(*keys_to_create):
            return user_func(*[key_lookup[k] for k in keys_to_create])
        timeout: Optional[float] = cast(ExpirationTimeCallable, expiration_time)() if expiration_time_is_callable else cast(Optional[float], expiration_time)
        result: Union[Sequence[ValuePayload], Mapping[KeyType, ValuePayload]]
        if asdict:

            def dict_create(*keys):
                d_values = creator(*keys)
                return [d_values.get(key_lookup[k], NO_VALUE) for k in keys]

            def wrap_cache_fn(value):
                if value is NO_VALUE:
                    return False
                elif not should_cache_fn:
                    return True
                else:
                    return should_cache_fn(value)
            result = self.get_or_create_multi(keys, dict_create, timeout, wrap_cache_fn)
            result = dict(((k, v) for k, v in zip(cache_keys, result) if v is not NO_VALUE))
        else:
            result = self.get_or_create_multi(keys, creator, timeout, should_cache_fn)
        return result

    def cache_decorator(user_func):
        key_generator = _function_multi_key_generator(namespace, user_func, to_str=to_str)

        def invalidate(*arg):
            keys = key_generator(*arg)
            self.delete_multi(keys)

        def set_(mapping):
            keys = list(mapping)
            gen_keys = key_generator(*keys)
            self.set_multi(dict(((gen_key, mapping[key]) for gen_key, key in zip(gen_keys, keys))))

        def get(*arg):
            keys = key_generator(*arg)
            return self.get_multi(keys)

        def refresh(*arg):
            keys = key_generator(*arg)
            values = user_func(*arg)
            if asdict:
                self.set_multi(dict(zip(keys, [values[a] for a in arg])))
                return values
            else:
                self.set_multi(dict(zip(keys, values)))
                return values
        user_func.set = set_
        user_func.invalidate = invalidate
        user_func.refresh = refresh
        user_func.get = get
        return decorate(user_func, partial(get_or_create_for_user_func, key_generator))
    return cache_decorator