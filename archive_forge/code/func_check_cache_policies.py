from __future__ import annotations
import time
import anyio
import inspect
import contextlib 
import functools
import hashlib
from lazyops.types.common import UpperStrEnum
from lazyops.utils import timed_cache
from lazyops.utils.helpers import create_background_task, fail_after
from lazyops.utils.lazy import lazy_import
from lazyops.utils.pooler import ThreadPooler
from lazyops.utils.lazy import get_function_name
from .compat import BaseModel, root_validator, get_pyd_dict
from .base import ENOVAL
from typing import Optional, Dict, Any, Callable, List, Union, TypeVar, Type, overload, TYPE_CHECKING
from aiokeydb.utils.logs import logger
from aiokeydb.utils.helpers import afail_after
def check_cache_policies(self, key: str, *args, **kwargs) -> None:
    """
        Runs the cache policies
        """
    if self.num_keys <= self.cache_max_size:
        return
    num_keys = self.num_keys
    if self.verbose:
        logger.info(f'[{self.cache_field}] Cache Max Size Reached: {num_keys}/{self.cache_max_size}. Running Cache Policy: {self.cache_max_size_policy}')
    if self.cache_max_size_policy == CachePolicy.LRU:
        timestamps = self._get('timestamps') or {}
        keys_to_delete = sorted(timestamps, key=timestamps.get)[:num_keys - self.cache_max_size]
        if key in keys_to_delete:
            keys_to_delete.remove(key)
        if self.verbose:
            logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
        self.clear(keys_to_delete)
        return
    if self.cache_max_size_policy == CachePolicy.LFU:
        key_hits = self._get('keyhits') or {}
        keys_to_delete = sorted(key_hits, key=key_hits.get)[:num_keys - self.cache_max_size]
        if key in keys_to_delete:
            keys_to_delete.remove(key)
        if self.verbose:
            logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
        self.clear(keys_to_delete)
        return
    if self.cache_max_size_policy == CachePolicy.FIFO:
        timestamps = self._get('timestamps') or {}
        keys_to_delete = sorted(timestamps, key=timestamps.get, reverse=True)[:num_keys - self.cache_max_size]
        if key in keys_to_delete:
            keys_to_delete.remove(key)
        if self.verbose:
            logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
        self.clear(keys_to_delete)
        return
    if self.cache_max_size_policy == CachePolicy.LIFO:
        timestamps = self._get('timestamps') or {}
        keys_to_delete = sorted(timestamps, key=timestamps.get)[:num_keys - self.cache_max_size]
        if key in keys_to_delete:
            keys_to_delete.remove(key)
        if self.verbose:
            logger.info(f'[{self.cache_field}] Deleting {len(keys_to_delete)} Keys: {keys_to_delete}')
        self.clear(keys_to_delete)
        return