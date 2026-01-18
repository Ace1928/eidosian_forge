import inspect
import operator
from io import StringIO
from webtest import TestApp
from pecan import make_app, expose, redirect, abort, rest, Request, Response
from pecan.hooks import (
from pecan.configuration import Config
from pecan.decorators import transactional, after_commit, after_rollback
from pecan.tests import PecanTestCase
def test_mixin_hooks(self):
    run_hook = []

    class HelperHook(PecanHook):
        priority = 2

        def before(self, state):
            run_hook.append('helper - before hook')
    helper_hook = HelperHook()

    class LastHook(PecanHook):
        priority = 200

        def before(self, state):
            run_hook.append('last - before hook')

    class SimpleHook(PecanHook):
        priority = 1

        def before(self, state):
            run_hook.append('simple - before hook')

    class HelperMixin(object):
        __hooks__ = [helper_hook]

    class LastMixin(object):
        __hooks__ = [LastHook()]

    class SubController(HookController, HelperMixin):
        __hooks__ = [LastHook()]

        @expose()
        def index(self):
            return 'This is sub controller!'

    class RootController(HookController, LastMixin):
        __hooks__ = [SimpleHook(), helper_hook]

        @expose()
        def index(self):
            run_hook.append('inside')
            return 'Hello, World!'
        sub = SubController()
    papp = make_app(RootController())
    app = TestApp(papp)
    response = app.get('/')
    assert response.status_int == 200
    assert response.body == b'Hello, World!'
    assert len(run_hook) == 4
    assert run_hook[0] == 'simple - before hook', run_hook[0]
    assert run_hook[1] == 'helper - before hook', run_hook[1]
    assert run_hook[2] == 'last - before hook', run_hook[2]
    assert run_hook[3] == 'inside', run_hook[3]
    run_hook = []
    response = app.get('/sub/')
    assert response.status_int == 200
    assert response.body == b'This is sub controller!'
    assert len(run_hook) == 4, run_hook
    assert run_hook[0] == 'simple - before hook', run_hook[0]
    assert run_hook[1] == 'helper - before hook', run_hook[1]
    assert run_hook[2] == 'last - before hook', run_hook[2]
    assert run_hook[3] == 'last - before hook', run_hook[3]