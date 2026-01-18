from __future__ import annotations
import gc
import atexit
import asyncio
import contextlib
import collections.abc
from lazyops.utils.lazy import lazy_import, get_keydb_enabled
from lazyops.utils.logs import logger, null_logger
from lazyops.utils.pooler import ThreadPooler
from typing import Any, Dict, Optional, Union, Iterable, List, Type, Set, Callable, Mapping, MutableMapping, Tuple, TypeVar, overload, TYPE_CHECKING
from .backends import LocalStatefulBackend, RedisStatefulBackend, StatefulBackendT
from .serializers import ObjectValue
from .addons import (
from .debug import get_autologger
def get_backend_class(self, **kwargs) -> Type['BackendT']:
    """
        Returns the Backend Class
        """
    if self.backend_type in RegisteredBackends:
        bt = RegisteredBackends[self.backend_type]
        return lazy_import(bt) if isinstance(bt, str) else bt
    if self.backend_type == 'auto':
        with contextlib.suppress(Exception):
            import kvdb
            if kvdb.is_available(url=kwargs.get('url')):
                from kvdb.components.persistence import KVDBStatefulBackend
                return KVDBStatefulBackend
        if get_keydb_enabled():
            return RedisStatefulBackend
        logger.warning('Defaulting to Local Stateful Backend')
        return LocalStatefulBackend
    raise NotImplementedError(f'Backend Type {self.backend_type} is not implemented')