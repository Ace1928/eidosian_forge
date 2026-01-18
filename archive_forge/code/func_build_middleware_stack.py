from __future__ import annotations
import sys
import typing
import warnings
from starlette.datastructures import State, URLPath
from starlette.middleware import Middleware, _MiddlewareClass
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.middleware.errors import ServerErrorMiddleware
from starlette.middleware.exceptions import ExceptionMiddleware
from starlette.requests import Request
from starlette.responses import Response
from starlette.routing import BaseRoute, Router
from starlette.types import ASGIApp, ExceptionHandler, Lifespan, Receive, Scope, Send
from starlette.websockets import WebSocket
def build_middleware_stack(self) -> ASGIApp:
    debug = self.debug
    error_handler = None
    exception_handlers: dict[typing.Any, typing.Callable[[Request, Exception], Response]] = {}
    for key, value in self.exception_handlers.items():
        if key in (500, Exception):
            error_handler = value
        else:
            exception_handlers[key] = value
    middleware = [Middleware(ServerErrorMiddleware, handler=error_handler, debug=debug)] + self.user_middleware + [Middleware(ExceptionMiddleware, handlers=exception_handlers, debug=debug)]
    app = self.router
    for cls, args, kwargs in reversed(middleware):
        app = cls(*args, app=app, **kwargs)
    return app