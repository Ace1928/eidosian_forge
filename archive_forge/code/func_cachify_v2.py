import typing
import logging
import threading
from aiokeydb.v1.lock import Lock
from aiokeydb.v1.connection import Encoder, ConnectionPool
from aiokeydb.v1.core import KeyDB, PubSub, Pipeline
from aiokeydb.v1.typing import Number, KeyT, AbsExpiryT, ExpiryT
from aiokeydb.v1.asyncio.lock import AsyncLock
from aiokeydb.v1.asyncio.core import AsyncKeyDB, AsyncPubSub, AsyncPipeline
from aiokeydb.v1.asyncio.connection import AsyncConnectionPool
from aiokeydb.v1.client.config import KeyDBSettings
from aiokeydb.v1.client.types import classproperty, KeyDBUri
from aiokeydb.v1.client.schemas.session import KeyDBSession
from aiokeydb.v1.client.serializers import SerializerType
@classmethod
def cachify_v2(cls, cache_ttl: int=None, typed: bool=False, cache_prefix: str=None, exclude: typing.List[str]=None, exclude_null: typing.Optional[bool]=False, exclude_return_types: typing.Optional[typing.List[type]]=None, exclude_return_objs: typing.Optional[typing.List[typing.Any]]=None, exclude_kwargs: typing.Optional[typing.List[str]]=None, include_cache_hit: typing.Optional[bool]=False, invalidate_cache_key: typing.Optional[str]=None, _no_cache: typing.Optional[bool]=False, _no_cache_kwargs: typing.Optional[typing.List[str]]=None, _no_cache_validator: typing.Optional[typing.Callable]=None, _func_name: typing.Optional[str]=None, _validate_requests: typing.Optional[bool]=True, _exclude_request_headers: typing.Optional[typing.Union[typing.List[str], bool]]=True, _cache_invalidator: typing.Optional[typing.Union[bool, typing.Callable]]=None, _invalidate_after_n_hits: typing.Optional[int]=None, _session: typing.Optional[str]=None, _cache_fallback: typing.Optional[bool]=True):
    """
        v2 of Cachify - detects whether a session is available and uses it
        if available, otherwise falls back to the `timed_cache` function
        """
    import contextlib
    from lazyops.utils import timed_cache
    _sess_ctx: 'KeyDBSession' = None

    def _get_sess_ctx():
        nonlocal _sess_ctx
        if _sess_ctx:
            return _sess_ctx
        with contextlib.suppress(Exception):
            _sess = cls.get_session(_session)
            if _sess.ping():
                _sess_ctx = _sess
        return _sess_ctx

    def wrapper_func(func):

        def inner_wrap(*args, **kwargs):
            sess_ctx = _get_sess_ctx()
            if sess_ctx is None:
                if _cache_fallback is True:
                    return timed_cache(secs=cache_ttl)(func)(*args, **kwargs)
                return func(*args, **kwargs)
            return sess_ctx.cachify(cache_ttl=cache_ttl, typed=typed, cache_prefix=cache_prefix, exclude=exclude, exclude_null=exclude_null, exclude_return_types=exclude_return_types, exclude_return_objs=exclude_return_objs, exclude_kwargs=exclude_kwargs, include_cache_hit=include_cache_hit, invalidate_cache_key=invalidate_cache_key, _no_cache=_no_cache, _no_cache_kwargs=_no_cache_kwargs, _no_cache_validator=_no_cache_validator, _func_name=_func_name, _validate_requests=_validate_requests, _exclude_request_headers=_exclude_request_headers, _cache_invalidator=_cache_invalidator, _invalidate_after_n_hits=_invalidate_after_n_hits)(func)(*args, **kwargs)
        return inner_wrap
    return wrapper_func