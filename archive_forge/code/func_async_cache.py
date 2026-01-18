import asyncio
import hashlib
import logging
import logging.config
import pickle
from functools import wraps
from typing import (
import aiokeydb
from aiokeydb import AsyncKeyDB, KeyDBError
from indedecorators import async_log_decorator
def async_cache(ttl: Optional[int]=None, key_prefix: str='', key_builder: Optional[Callable[..., CacheKeyType]]=None, cache_instance: CacheInstanceType=None, retry_limit: int=CACHE_RETRY_LIMIT, retry_delay: int=CACHE_RETRY_DELAY) -> Callable[[DecoratedCallable], DecoratedCallable]:
    """
    A decorator that caches the result of an asynchronous function using KeyDB.

    Args:
        ttl (Optional[int], optional): Time-to-live (TTL) for the cached result in seconds. Defaults to None.
        key_prefix (str, optional): Prefix to be added to the cache key. Defaults to "".
        key_builder (Optional[Callable[..., CacheKeyType]], optional): A function to build a custom cache key. Defaults to None.
        cache_instance (CacheInstanceType, optional): The KeyDB cache instance to use. Defaults to None.
        retry_limit (int, optional): Maximum number of retries for cache operations. Defaults to CACHE_RETRY_LIMIT.
        retry_delay (int, optional): Delay in seconds between retries for cache operations. Defaults to CACHE_RETRY_DELAY.

    Returns:
        Callable[[DecoratedCallable], DecoratedCallable]: The decorated function.
    """

    def decorator(func: DecoratedCallable) -> Awaitable[Any]:

        @wraps(func)
        @async_log_decorator
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            nonlocal cache_instance
            if cache_instance is None:
                cache_instance = global_cache
            if cache_instance is None:
                logger.warning('No cache instance available. Executing the function without caching.')
                return await func(*args, **kwargs)
            cache_key: CacheKeyType = key_builder(*args, **kwargs) if key_builder else f'{key_prefix}:{func.__name__}:{args}:{kwargs}'
            cache_key_hash: str = hashlib.md5(str(cache_key).encode()).hexdigest()
            for attempt in range(retry_limit):
                try:
                    cached_result: Optional[SerializedCacheData] = await cache_instance.get(cache_key_hash)
                    if cached_result is not None:
                        logger.info(f'Returning cached result for {func.__name__} with key: {cache_key}')
                        return cast(CacheValueType, pickle.loads(cached_result))
                    result: CacheValueType = await func(*args, **kwargs)
                    serialized_result: SerializedCacheData = pickle.dumps(result)
                    await cache_instance.set(cache_key_hash, serialized_result, ex=ttl)
                    logger.info(f'Cached result for {func.__name__} with key: {cache_key}')
                    return result
                except Exception as e:
                    logger.exception(f'KeyDB error occurred while accessing cache for {func.__name__}: {e}')
                    if attempt < retry_limit - 1:
                        logger.info(f'Retrying cache operation for {func.__name__} in {retry_delay} seconds...')
                        await asyncio.sleep(retry_delay)
                    else:
                        logger.error(f'Exceeded retry limit for cache operations in {func.__name__}. Executing without caching.')
                        return await func(*args, **kwargs)
        return wrapper
    return decorator