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
def fallback_async_wrapper(func: FT, session: 'KeyDBSession', _kwargs: CachifyKwargs) -> FT:
    """
    [Async] Handles the fallback wrapper
    """
    _sess_ctx: Optional['KeyDBSession'] = None

    async def _get_sess():
        nonlocal _sess_ctx
        if _sess_ctx is None:
            with contextlib.suppress(Exception):
                async with afail_after(1.0):
                    if await session.async_client.ping():
                        _sess_ctx = session
            if _kwargs.verbose and _sess_ctx is None:
                logger.error('Could not connect to KeyDB')
        return _sess_ctx

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        """
        The wrapper for cachify
        """
        _sess = await _get_sess()
        if _sess is None:
            with contextlib.suppress(Exception):
                return await timed_cache(secs=_kwargs.ttl)
            return await func(*args, **kwargs)
        return await cachify_async(_sess, _kwargs)(func)(*args, **kwargs)

    async def clear(keys: Optional[Union[str, List[str]]]=None, **kwargs) -> Optional[int]:
        """
        Clears the cache
        """
        return await _kwargs.aclear(keys=keys)

    async def num_hits(*args, **kwargs) -> int:
        """
        Returns the number of hits
        """
        return await _kwargs.anum_hits

    async def num_keys(**kwargs) -> int:
        """
        Returns the number of keys
        """
        return await _kwargs.anum_keys

    async def cache_keys(**kwargs) -> List[str]:
        """
        Returns the keys
        """
        return await _kwargs.acache_keys

    async def cache_values(**kwargs) -> List[Any]:
        """
        Returns the values
        """
        return await _kwargs.acache_values

    async def cache_items(**kwargs) -> Dict[str, Any]:
        """
        Returns the items
        """
        return await _kwargs.acache_items

    async def invalidate_key(key: str, **kwargs) -> int:
        """
        Invalidates the cache
        """
        return await _kwargs.ainvalidate_cache(key)

    async def cache_timestamps(**kwargs) -> Dict[str, float]:
        """
        Returns the timestamps
        """
        return await _kwargs.acache_timestamps

    async def cache_keyhits(**kwargs) -> Dict[str, int]:
        """
        Returns the keyhits
        """
        return await _kwargs.acache_keyhits

    async def cache_policy(**kwargs) -> Dict[str, Union[int, CachePolicy]]:
        """
        Returns the cache policy
        """
        return {'max_size': _kwargs.cache_max_size, 'max_size_policy': _kwargs.cache_max_size_policy}

    async def cache_config(**kwargs) -> Dict[str, Any]:
        """
        Returns the cache config
        """
        values = get_pyd_dict(_kwargs, exclude={'session'})
        for k, v in values.items():
            if callable(v):
                values[k] = get_function_name(v)
        return values

    async def cache_info(**kwargs) -> Dict[str, Any]:
        """
        Returns the info for the cache
        """
        return await _kwargs.acache_info

    async def cache_update(**kwargs) -> Dict[str, Any]:
        """
        Updates the cache config
        """
        _kwargs.update(**kwargs)
        return await cache_config(**kwargs)
    wrapper.clear = clear
    wrapper.num_hits = num_hits
    wrapper.num_keys = num_keys
    wrapper.cache_keys = cache_keys
    wrapper.cache_values = cache_values
    wrapper.cache_items = cache_items
    wrapper.invalidate_key = invalidate_key
    wrapper.cache_timestamps = cache_timestamps
    wrapper.cache_keyhits = cache_keyhits
    wrapper.cache_policy = cache_policy
    wrapper.cache_config = cache_config
    wrapper.cache_info = cache_info
    wrapper.cache_update = cache_update
    return wrapper