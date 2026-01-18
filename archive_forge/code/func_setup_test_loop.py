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
def setup_test_loop(loop_factory: _LOOP_FACTORY=asyncio.new_event_loop) -> asyncio.AbstractEventLoop:
    """Create and return an asyncio.BaseEventLoop instance.

    The caller should also call teardown_test_loop,
    once they are done with the loop.
    """
    loop = loop_factory()
    asyncio.set_event_loop(loop)
    return loop