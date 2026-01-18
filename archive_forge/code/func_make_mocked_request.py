import asyncio
import contextlib
import gc
import inspect
import ipaddress
import os
import socket
import sys
import warnings
from abc import ABC, abstractmethod
from types import TracebackType
from typing import (
from unittest import IsolatedAsyncioTestCase, mock
from aiosignal import Signal
from multidict import CIMultiDict, CIMultiDictProxy
from yarl import URL
import aiohttp
from aiohttp.client import _RequestContextManager, _WSRequestContextManager
from . import ClientSession, hdrs
from .abc import AbstractCookieJar
from .client_reqrep import ClientResponse
from .client_ws import ClientWebSocketResponse
from .helpers import sentinel
from .http import HttpVersion, RawRequestMessage
from .typedefs import StrOrURL
from .web import (
from .web_protocol import _RequestHandler
def make_mocked_request(method: str, path: str, headers: Any=None, *, match_info: Any=sentinel, version: HttpVersion=HttpVersion(1, 1), closing: bool=False, app: Any=None, writer: Any=sentinel, protocol: Any=sentinel, transport: Any=sentinel, payload: Any=sentinel, sslcontext: Optional[SSLContext]=None, client_max_size: int=1024 ** 2, loop: Any=...) -> Request:
    """Creates mocked web.Request testing purposes.

    Useful in unit tests, when spinning full web server is overkill or
    specific conditions and errors are hard to trigger.
    """
    task = mock.Mock()
    if loop is ...:
        loop = mock.Mock()
        loop.create_future.return_value = ()
    if version < HttpVersion(1, 1):
        closing = True
    if headers:
        headers = CIMultiDictProxy(CIMultiDict(headers))
        raw_hdrs = tuple(((k.encode('utf-8'), v.encode('utf-8')) for k, v in headers.items()))
    else:
        headers = CIMultiDictProxy(CIMultiDict())
        raw_hdrs = ()
    chunked = 'chunked' in headers.get(hdrs.TRANSFER_ENCODING, '').lower()
    message = RawRequestMessage(method, path, version, headers, raw_hdrs, closing, None, False, chunked, URL(path))
    if app is None:
        app = _create_app_mock()
    if transport is sentinel:
        transport = _create_transport(sslcontext)
    if protocol is sentinel:
        protocol = mock.Mock()
        protocol.transport = transport
    if writer is sentinel:
        writer = mock.Mock()
        writer.write_headers = make_mocked_coro(None)
        writer.write = make_mocked_coro(None)
        writer.write_eof = make_mocked_coro(None)
        writer.drain = make_mocked_coro(None)
        writer.transport = transport
    protocol.transport = transport
    protocol.writer = writer
    if payload is sentinel:
        payload = mock.Mock()
    req = Request(message, payload, protocol, writer, task, loop, client_max_size=client_max_size)
    match_info = UrlMappingMatchInfo({} if match_info is sentinel else match_info, mock.Mock())
    match_info.add_app(app)
    req._match_info = match_info
    return req