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
def test_locals_are_not_used(self):
    with mock.patch('threading.local', side_effect=AssertionError()):
        app = TestApp(Pecan(self.root(), use_context_locals=False))
        r = app.get('/')
        assert r.status_int == 200
        assert r.body == b'Hello, World!'
        self.assertRaises(AssertionError, Pecan, self.root)