import typing
from starlette.authentication import (
from starlette.requests import HTTPConnection
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send
@staticmethod
def default_on_error(conn: HTTPConnection, exc: Exception) -> Response:
    return PlainTextResponse(str(exc), status_code=400)