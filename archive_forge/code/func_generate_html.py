import html
import inspect
import traceback
import typing
from starlette._utils import is_async_callable
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request
from starlette.responses import HTMLResponse, PlainTextResponse, Response
from starlette.types import ASGIApp, Message, Receive, Scope, Send
def generate_html(self, exc: Exception, limit: int=7) -> str:
    traceback_obj = traceback.TracebackException.from_exception(exc, capture_locals=True)
    exc_html = ''
    is_collapsed = False
    exc_traceback = exc.__traceback__
    if exc_traceback is not None:
        frames = inspect.getinnerframes(exc_traceback, limit)
        for frame in reversed(frames):
            exc_html += self.generate_frame_html(frame, is_collapsed)
            is_collapsed = True
    error = f'{html.escape(traceback_obj.exc_type.__name__)}: {html.escape(str(traceback_obj))}'
    return TEMPLATE.format(styles=STYLES, js=JS, error=error, exc_html=exc_html)