import html
import inspect
import traceback
import typing
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
def generate_frame_html(self, frame: inspect.FrameInfo, is_collapsed: bool) -> str:
    code_context = ''.join((self.format_line(index, line, frame.lineno, frame.index) for index, line in enumerate(frame.code_context or [])))
    values = {'frame_filename': html.escape(frame.filename), 'frame_lineno': frame.lineno, 'frame_name': html.escape(frame.function), 'code_context': code_context, 'collapsed': 'collapsed' if is_collapsed else '', 'collapse_button': '+' if is_collapsed else '&#8210;'}
    return FRAME_TEMPLATE.format(**values)