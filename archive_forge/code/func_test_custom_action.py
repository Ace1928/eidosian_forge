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
def test_custom_action(self):
    r = self.app_.get('/things/3/edit')
    assert r.status_int == 200
    assert r.body == b'EDIT three'