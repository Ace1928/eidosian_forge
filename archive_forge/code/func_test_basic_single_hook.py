import time
from json import dumps, loads
import warnings
from unittest import mock
from webtest import TestApp
import webob
from pecan import Pecan, expose, abort, Request, Response
from pecan.rest import RestController
from pecan.hooks import PecanHook, HookController
from pecan.tests import PecanTestCase
def test_basic_single_hook(self):
    run_hook = []

    class RootController(object):

        @expose()
        def index(self, req, resp):
            run_hook.append('inside')
            return 'Hello, World!'

    class SimpleHook(PecanHook):

        def on_route(self, state):
            run_hook.append('on_route')

        def before(self, state):
            run_hook.append('before')

        def after(self, state):
            run_hook.append('after')

        def on_error(self, state, e):
            run_hook.append('error')
    app = TestApp(Pecan(RootController(), hooks=[SimpleHook()], use_context_locals=False))
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert len(run_hook) == 4
    assert run_hook[0] == 'on_route'
    assert run_hook[1] == 'before'
    assert run_hook[2] == 'inside'
    assert run_hook[3] == 'after'