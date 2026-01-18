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
def replace_params(path: str, param_convertors: dict[str, Convertor[typing.Any]], path_params: dict[str, str]) -> tuple[str, dict[str, str]]:
    for key, value in list(path_params.items()):
        if '{' + key + '}' in path:
            convertor = param_convertors[key]
            value = convertor.to_string(value)
            path = path.replace('{' + key + '}', value)
            path_params.pop(key)
    return (path, path_params)