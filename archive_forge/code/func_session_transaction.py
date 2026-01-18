from __future__ import annotations
import importlib.metadata
import typing as t
from contextlib import contextmanager
from contextlib import ExitStack
from copy import copy
from types import TracebackType
from urllib.parse import urlsplit
import werkzeug.test
from click.testing import CliRunner
from werkzeug.test import Client
from werkzeug.wrappers import Request as BaseRequest
from .cli import ScriptInfo
from .sessions import SessionMixin
@contextmanager
def session_transaction(self, *args: t.Any, **kwargs: t.Any) -> t.Iterator[SessionMixin]:
    """When used in combination with a ``with`` statement this opens a
        session transaction.  This can be used to modify the session that
        the test client uses.  Once the ``with`` block is left the session is
        stored back.

        ::

            with client.session_transaction() as session:
                session['value'] = 42

        Internally this is implemented by going through a temporary test
        request context and since session handling could depend on
        request variables this function accepts the same arguments as
        :meth:`~flask.Flask.test_request_context` which are directly
        passed through.
        """
    if self._cookies is None:
        raise TypeError("Cookies are disabled. Create a client with 'use_cookies=True'.")
    app = self.application
    ctx = app.test_request_context(*args, **kwargs)
    self._add_cookies_to_wsgi(ctx.request.environ)
    with ctx:
        sess = app.session_interface.open_session(app, ctx.request)
    if sess is None:
        raise RuntimeError('Session backend did not open a session.')
    yield sess
    resp = app.response_class()
    if app.session_interface.is_null_session(sess):
        return
    with ctx:
        app.session_interface.save_session(app, sess, resp)
    self._update_cookies_from_response(ctx.request.host.partition(':')[0], ctx.request.path, resp.headers.getlist('Set-Cookie'))