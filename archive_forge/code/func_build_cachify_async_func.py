import sys
import time
import anyio
import typing
import logging
import asyncio
import functools
import contextlib
from pydantic import BaseModel
from pydantic.types import ByteSize
from aiokeydb.v2.typing import Number, KeyT, ExpiryT, AbsExpiryT, PatternT
from aiokeydb.v2.lock import Lock, AsyncLock
from aiokeydb.v2.core import KeyDB, PubSub, Pipeline, PipelineT, PubSubT
from aiokeydb.v2.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline, AsyncPipelineT, AsyncPubSubT
from aiokeydb.v2.connection import Encoder, ConnectionPool, AsyncConnectionPool
from aiokeydb.v2.exceptions import (
from aiokeydb.v2.types import KeyDBUri, ENOVAL
from aiokeydb.v2.configs import KeyDBSettings, settings as default_settings
from aiokeydb.v2.utils import full_name, args_to_key
from aiokeydb.v2.utils.helpers import create_retryable_client
from aiokeydb.v2.serializers import BaseSerializer
from inspect import iscoroutinefunction
def build_cachify_async_func(session: 'KeyDBSession', func: typing.Callable, base: str, cache_ttl: int=None, typed: bool=False, cache_prefix: str=None, exclude: typing.List[str]=None, exclude_null: typing.Optional[bool]=False, exclude_return_types: typing.Optional[typing.List[type]]=None, exclude_return_objs: typing.Optional[typing.List[typing.Any]]=None, exclude_kwargs: typing.Optional[typing.List[str]]=None, include_cache_hit: typing.Optional[bool]=False, invalidate_cache_key: typing.Optional[str]=None, _no_cache: typing.Optional[bool]=False, _no_cache_kwargs: typing.Optional[typing.List[str]]=None, _no_cache_validator: typing.Optional[typing.Callable]=None, _validate_requests: typing.Optional[bool]=True, _exclude_request_headers: typing.Optional[typing.Union[typing.List[str], bool]]=True, _cache_invalidator: typing.Optional[typing.Union[bool, typing.Callable]]=None, _invalidate_after_n_hits: typing.Optional[int]=None, _cache_timeout: typing.Optional[float]=5.0, **kwargs) -> typing.Callable:
    """
    Builds a cachify function
    """

    @functools.wraps(func)
    async def wrapper(*args, **kwargs):
        """Wrapper for callable to cache arguments and return values."""
        __invalidate_cache = kwargs.pop(invalidate_cache_key, False) if invalidate_cache_key else False
        if __invalidate_cache is None:
            __invalidate_cache = False
        if session.cache_enabled is False:
            if include_cache_hit:
                return (await func(*args, **kwargs), False)
            return await func(*args, **kwargs)
        if (session.state.cache_max_attempts and session.state.cache_max_attempts > 0) and session.state.cache_failed_attempts >= session.state.cache_max_attempts:
            if include_cache_hit:
                return (await func(*args, **kwargs), False)
            return await func(*args, **kwargs)
        if _no_cache:
            if include_cache_hit:
                return (await func(*args, **kwargs), False)
            return await func(*args, **kwargs)
        if _no_cache_kwargs and any((kwargs.get(k) for k in _no_cache_kwargs)):
            if include_cache_hit:
                return (await func(*args, **kwargs), False)
            return await func(*args, **kwargs)
        if _no_cache_validator:
            if iscoroutinefunction(_no_cache_validator):
                if await _no_cache_validator(*args, **kwargs):
                    if include_cache_hit:
                        return (await func(*args, **kwargs), False)
                    return await func(*args, **kwargs)
            elif _no_cache_validator(*args, **kwargs):
                if include_cache_hit:
                    return (await func(*args, **kwargs), False)
                return await func(*args, **kwargs)
        keybuilder_kwargs = kwargs.copy()
        if _validate_requests:
            copy_kwargs = kwargs.copy()
            request = copy_kwargs.pop('request', None)
            headers: typing.Dict[str, str] = kwargs.pop('headers', getattr(request, 'headers', kwargs.pop('headers', None)))
            if headers is not None:
                check_headers = {k.lower(): v for k, v in headers.items()}
                for key in {'cache-control', 'x-cache-control', 'x-no-cache'}:
                    if check_headers.get(key, '') in {'no-store', 'no-cache', 'true'}:
                        if include_cache_hit:
                            return (await func(*args, **kwargs), False)
                        return await func(*args, **kwargs)
        if exclude_kwargs:
            for key in exclude_kwargs:
                _ = keybuilder_kwargs.pop(key, None)
        key = wrapper.__cache_key__(*args, **keybuilder_kwargs)
        _invalidate_key = __invalidate_cache
        if not _invalidate_key and _cache_invalidator:
            if isinstance(_cache_invalidator, bool):
                _invalidate_key = _cache_invalidator
            elif iscoroutinefunction(_cache_invalidator):
                _invalidate_key = await _cache_invalidator(*args, _cache_key=key, **kwargs)
            else:
                _invalidate_key = _cache_invalidator(*args, _cache_key=key, **kwargs)
        if _invalidate_after_n_hits:
            _hits_key = f'{key}:hits'
            _num_hits = 0
            with contextlib.suppress(Exception):
                with anyio.fail_after(_cache_timeout):
                    _num_hits = await session.async_get(_hits_key)
                    if _num_hits:
                        _num_hits = int(_num_hits)
            if _num_hits and _num_hits > _invalidate_after_n_hits:
                _invalidate_key = True
                with contextlib.suppress(Exception):
                    with anyio.fail_after(_cache_timeout):
                        await session.async_delete(key)
        if _invalidate_key:
            if session.settings.debug_enabled:
                logger.info(f'[{session.name}] Invalidating cache key: {key}')
            with contextlib.suppress(Exception):
                with anyio.fail_after(_cache_timeout):
                    await session.async_delete(key)
        try:
            with anyio.fail_after(_cache_timeout):
                result = await session.async_get(key, default=ENOVAL)
        except TimeoutError:
            result = ENOVAL
            logger.warning(f'[{session.name}] Calling GET on async KeyDB timed out. Cached function: {base}')
            session.state.cache_failed_attempts += 1
        except Exception as e:
            result = ENOVAL
            logger.warning(f'[{session.name}] Calling GET on async KeyDB failed. Cached function: {base}: {e}')
            session.state.cache_failed_attempts += 1
        is_cache_hit = result is not ENOVAL
        if not is_cache_hit:
            result = await func(*args, **kwargs)
            if exclude_null and result is None:
                return (result, False) if include_cache_hit else result
            if exclude_return_types and isinstance(result, tuple(exclude_return_types)):
                return (result, False) if include_cache_hit else result
            if exclude_return_objs and issubclass(type(result), tuple(exclude_return_objs)):
                return (result, False) if include_cache_hit else result
            if cache_ttl is None or cache_ttl > 0:
                try:
                    with anyio.fail_after(_cache_timeout):
                        await session.async_set(key, result, ex=cache_ttl)
                except TimeoutError:
                    logger.error(f'[{session.name}] Calling SET on async KeyDB timed out. Cached function: {base}')
                    session.state.cache_failed_attempts += 1
                except Exception as e:
                    logger.error(f'[{session.name}] Calling SET on async KeyDB failed. Cached function: {base}: {e}')
                    session.state.cache_failed_attempts += 1
        elif _invalidate_after_n_hits:
            with contextlib.suppress(Exception):
                with anyio.fail_after(_cache_timeout):
                    await session.async_incr(_hits_key)
        return (result, is_cache_hit) if include_cache_hit else result

    def __cache_key__(*args, **kwargs):
        """Make key for cache given function arguments."""
        return f'{cache_prefix}_{args_to_key(base, args, kwargs, typed, exclude)}'
    wrapper.__cache_key__ = __cache_key__
    return wrapper