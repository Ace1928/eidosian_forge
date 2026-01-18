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
def cachify_sync(sess: 'KeyDBSession', _kwargs: CachifyKwargs) -> FT:
    """
    Handles the sync caching
    """
    _kwargs.session = sess
    _kwargs.is_async = False

    def decorator(func):

        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            _kwargs.build_hash_name(func, *args, **kwargs)
            _kwargs.validate_is_class_method(func)
            _kwargs.run_post_init_hook(func, *args, **kwargs)
            if not _kwargs.should_cache(*args, **kwargs):
                if _kwargs.super_verbose:
                    logger.info('Not Caching', prefix=_kwargs.cache_field, colored=True)
                return func(*args, **kwargs)
            if _kwargs.should_bypass(*args, **kwargs):
                if _kwargs.super_verbose:
                    logger.info('Bypassing', prefix=_kwargs.cache_field, colored=True)
                return func(*args, **kwargs)
            cache_key = wrapper.__cache_key__(*args, **kwargs)
            if _kwargs.should_invalidate(*args, **kwargs):
                if _kwargs.verbose:
                    logger.info('Invalidating', prefix=f'{_kwargs.cache_field}:{cache_key}', colored=True)
                _kwargs.invalidate_cache(cache_key)
            value = _kwargs.retrieve(cache_key, *args, **kwargs)
            if value == ENOVAL:
                try:
                    value = func(*args, **kwargs)
                    if _kwargs.should_cache_value(value):
                        if _kwargs.super_verbose:
                            logger.info('Caching Value', prefix=f'{_kwargs.cache_field}:{cache_key}', colored=True)
                        _kwargs.set(cache_key, value, *args, **kwargs)
                    if _kwargs.super_verbose:
                        logger.info('Cache Miss', prefix=f'{_kwargs.cache_field}:{cache_key}', colored=True)
                    _kwargs.run_post_call_hook(value, *args, is_hit=False, **kwargs)
                    return value
                except Exception as e:
                    if _kwargs.verbose:
                        logger.trace(f'[{_kwargs.cache_field}:{cache_key}] Exception', error=e)
                    if _kwargs.raise_exceptions:
                        raise e
                    return None
            if _kwargs.super_verbose:
                logger.info('Cache Hit', prefix=f'{_kwargs.cache_field}:{cache_key}', colored=True)
            _kwargs.run_post_call_hook(value, *args, is_hit=True, **kwargs)
            return value

        def __cache_key__(*args, **kwargs) -> str:
            """
            Returns the cache key
            """
            return _kwargs.build_hash_key(*args, **kwargs)
        wrapper.__cache_key__ = __cache_key__
        return wrapper
    return decorator