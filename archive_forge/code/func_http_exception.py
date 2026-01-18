import typing
from starlette._exception_handler import (
from starlette.exceptions import HTTPException, WebSocketException
from starlette.requests import Request
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
from starlette.websockets import WebSocket
def http_exception(self, request: Request, exc: Exception) -> Response:
    assert isinstance(exc, HTTPException)
    if exc.status_code in {204, 304}:
        return Response(status_code=exc.status_code, headers=exc.headers)
    return PlainTextResponse(exc.detail, status_code=exc.status_code, headers=exc.headers)