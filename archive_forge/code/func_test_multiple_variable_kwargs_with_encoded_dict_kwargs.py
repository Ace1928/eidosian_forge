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
def test_multiple_variable_kwargs_with_encoded_dict_kwargs(self):
    r = self.app_.post('/variable_kwargs', {'id': 'Three%21', 'dummy': 'This%20is%20a%20test'})
    assert r.status_int == 200
    result = b'variable_kwargs: dummy=This%20is%20a%20test, id=Three%21'
    assert r.body == result