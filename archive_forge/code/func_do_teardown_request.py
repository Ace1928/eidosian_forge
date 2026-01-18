from __future__ import annotations
import collections.abc as cabc
import os
import sys
import typing as t
import weakref
from datetime import timedelta
from inspect import iscoroutinefunction
from itertools import chain
from types import TracebackType
from urllib.parse import quote as _url_quote
import click
from werkzeug.datastructures import Headers
from werkzeug.datastructures import ImmutableDict
from werkzeug.exceptions import BadRequestKeyError
from werkzeug.exceptions import HTTPException
from werkzeug.exceptions import InternalServerError
from werkzeug.routing import BuildError
from werkzeug.routing import MapAdapter
from werkzeug.routing import RequestRedirect
from werkzeug.routing import RoutingException
from werkzeug.routing import Rule
from werkzeug.serving import is_running_from_reloader
from werkzeug.wrappers import Response as BaseResponse
from . import cli
from . import typing as ft
from .ctx import AppContext
from .ctx import RequestContext
from .globals import _cv_app
from .globals import _cv_request
from .globals import current_app
from .globals import g
from .globals import request
from .globals import request_ctx
from .globals import session
from .helpers import get_debug_flag
from .helpers import get_flashed_messages
from .helpers import get_load_dotenv
from .helpers import send_from_directory
from .sansio.app import App
from .sansio.scaffold import _sentinel
from .sessions import SecureCookieSessionInterface
from .sessions import SessionInterface
from .signals import appcontext_tearing_down
from .signals import got_request_exception
from .signals import request_finished
from .signals import request_started
from .signals import request_tearing_down
from .templating import Environment
from .wrappers import Request
from .wrappers import Response
def do_teardown_request(self, exc: BaseException | None=_sentinel) -> None:
    """Called after the request is dispatched and the response is
        returned, right before the request context is popped.

        This calls all functions decorated with
        :meth:`teardown_request`, and :meth:`Blueprint.teardown_request`
        if a blueprint handled the request. Finally, the
        :data:`request_tearing_down` signal is sent.

        This is called by
        :meth:`RequestContext.pop() <flask.ctx.RequestContext.pop>`,
        which may be delayed during testing to maintain access to
        resources.

        :param exc: An unhandled exception raised while dispatching the
            request. Detected from the current exception information if
            not passed. Passed to each teardown function.

        .. versionchanged:: 0.9
            Added the ``exc`` argument.
        """
    if exc is _sentinel:
        exc = sys.exc_info()[1]
    for name in chain(request.blueprints, (None,)):
        if name in self.teardown_request_funcs:
            for func in reversed(self.teardown_request_funcs[name]):
                self.ensure_sync(func)(exc)
    request_tearing_down.send(self, _async_wrapper=self.ensure_sync, exc=exc)