import functools
import re
import typing
from starlette.datastructures import Headers, MutableHeaders
from starlette.responses import PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
def is_allowed_origin(self, origin: str) -> bool:
    if self.allow_all_origins:
        return True
    if self.allow_origin_regex is not None and self.allow_origin_regex.fullmatch(origin):
        return True
    return origin in self.allow_origins