from __future__ import annotations
import copy
import math
import operator
import typing as t
from contextvars import ContextVar
from functools import partial
from functools import update_wrapper
from operator import attrgetter
from .wsgi import ClosingIterator
def make_middleware(self, app: WSGIApplication) -> WSGIApplication:
    """Wrap a WSGI application so that local data is released
        automatically after the response has been sent for a request.
        """

    def application(environ: WSGIEnvironment, start_response: StartResponse) -> t.Iterable[bytes]:
        return ClosingIterator(app(environ, start_response), self.cleanup)
    return application