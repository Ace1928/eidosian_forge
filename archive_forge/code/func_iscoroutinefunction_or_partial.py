from __future__ import annotations
import contextlib
import functools
import inspect
import re
import traceback
import types
import typing
import warnings
from contextlib import asynccontextmanager
from enum import Enum
from starlette._exception_handler import wrap_app_handling_exceptions
from starlette._utils import get_route_path, is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.convertors import CONVERTOR_TYPES, Convertor
from starlette.datastructures import URL, Headers, URLPath
from starlette.exceptions import HTTPException
from starlette.middleware import Middleware
from starlette.requests import Request
from starlette.responses import PlainTextResponse, RedirectResponse, Response
from starlette.types import ASGIApp, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket, WebSocketClose
def iscoroutinefunction_or_partial(obj: typing.Any) -> bool:
    """
    Correctly determines if an object is a coroutine function,
    including those wrapped in functools.partial objects.
    """
    warnings.warn('iscoroutinefunction_or_partial is deprecated, and will be removed in a future release.', DeprecationWarning)
    while isinstance(obj, functools.partial):
        obj = obj.func
    return inspect.iscoroutinefunction(obj)