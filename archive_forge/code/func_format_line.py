import html
import inspect
import traceback
import typing
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
def format_line(self, index: int, line: str, frame_lineno: int, frame_index: int) -> str:
    values = {'line': html.escape(line).replace(' ', '&nbsp'), 'lineno': frame_lineno - frame_index + index}
    if index != frame_index:
        return LINE.format(**values)
    return CENTER_LINE.format(**values)