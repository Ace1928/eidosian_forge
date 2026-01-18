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
def test_delete_with_method_parameter(self):
    r = self.app_.get('/things/3?_method=DELETE', status=405)
    assert r.status_int == 405