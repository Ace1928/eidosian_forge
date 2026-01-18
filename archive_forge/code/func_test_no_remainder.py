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
def test_no_remainder(self):
    try:
        r = self.app_.get('/eater')
        assert r.status_int != 200
    except Exception as ex:
        assert type(ex) == TypeError
        assert ex.args[0] in ('eater() takes exactly 2 arguments (1 given)', "eater() missing 1 required positional argument: 'id'", "TestControllerArguments.app_.<locals>.RootController.eater() missing 1 required positional argument: 'id'")