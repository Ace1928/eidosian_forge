import functools
import re
import typing
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
@staticmethod
def allow_explicit_origin(headers: MutableHeaders, origin: str) -> None:
    headers['Access-Control-Allow-Origin'] = origin
    headers.add_vary_header('Origin')