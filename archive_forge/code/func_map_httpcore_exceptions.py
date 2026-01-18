from __future__ import annotations
import contextlib
import typing
from types import TracebackType
import httpcore
from .._config import DEFAULT_LIMITS, Limits, Proxy, create_ssl_context
from .._exceptions import (
from .._models import Request, Response
from .._types import AsyncByteStream, CertTypes, ProxyTypes, SyncByteStream, VerifyTypes
from .._urls import URL
from .base import AsyncBaseTransport, BaseTransport
@contextlib.contextmanager
def map_httpcore_exceptions() -> typing.Iterator[None]:
    try:
        yield
    except Exception as exc:
        mapped_exc = None
        for from_exc, to_exc in HTTPCORE_EXC_MAP.items():
            if not isinstance(exc, from_exc):
                continue
            if mapped_exc is None or issubclass(to_exc, mapped_exc):
                mapped_exc = to_exc
        if mapped_exc is None:
            raise
        message = str(exc)
        raise mapped_exc(message) from exc