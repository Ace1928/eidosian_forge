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
def get_child_kwargs(self, **kwargs) -> Dict[str, Any]:
    """
        Returns the Child Kwargs
        """
    base_kwargs = self._kwargs.copy()
    if kwargs:
        base_kwargs.update(kwargs)
    if 'settings' not in base_kwargs:
        base_kwargs['settings'] = self.settings
    if 'name' not in base_kwargs:
        base_kwargs['name'] = self.name
    if 'backend_type' not in base_kwargs and 'backend' not in base_kwargs:
        base_kwargs['backend'] = self.base_class
    if 'async_enabled' not in base_kwargs:
        base_kwargs['async_enabled'] = self.base.async_enabled
    return base_kwargs