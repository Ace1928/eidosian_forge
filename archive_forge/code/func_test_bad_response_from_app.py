import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_bad_response_from_app(self):
    """When exceptions are raised the hook deals with them properly"""
    _stdout = StringIO()

    class RootController(object):

        @expose()
        def index(self):
            return 'Hello, World!'
    app = TestApp(make_app(RootController(), hooks=lambda: [RequestViewerHook(writer=_stdout)]))
    response = app.get('/404', expect_errors=True)
    out = _stdout.getvalue()
    assert response.status_int == 404
    assert 'path' in out
    assert 'method' in out
    assert 'status' in out
    assert 'method' in out
    assert 'params' in out
    assert 'hooks' in out
    assert '404 Not Found' in out
    assert "['RequestViewerHook']" in out
    assert '/' in out