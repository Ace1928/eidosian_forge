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
def get_value_metadata(self, key: KeyType, expiration_time: Optional[float]=None, ignore_expiration: bool=False) -> Optional[CachedValue]:
    """Return the :class:`.CachedValue` object directly from the cache.

        This is the enclosing datastructure that includes the value as well as
        the metadata, including the timestamp when the value was cached.
        Convenience accessors on :class:`.CachedValue` also provide for common
        data such as :attr:`.CachedValue.cached_time` and
        :attr:`.CachedValue.age`.


        .. versionadded:: 1.3. Added :meth:`.CacheRegion.get_value_metadata`
        """
    cache_value = self._get_cache_value(key, expiration_time, ignore_expiration)
    if cache_value is NO_VALUE:
        return None
    else:
        if TYPE_CHECKING:
            assert isinstance(cache_value, CachedValue)
        return cache_value