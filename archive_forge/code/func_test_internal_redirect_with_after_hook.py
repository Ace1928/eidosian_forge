import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_internal_redirect_with_after_hook(self):
    run_hook = []

    class RootController(object):

        @expose()
        def internal(self):
            redirect('/testing', internal=True)

        @expose()
        def testing(self):
            return 'it worked!'

    class SimpleHook(PecanHook):

        def after(self, state):
            run_hook.append('after')
    app = TestApp(make_app(RootController(), hooks=[SimpleHook()]))
    response = app.get('/internal')
    assert response.body == b'it worked!'
    assert len(run_hook) == 1