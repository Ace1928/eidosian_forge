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
@expose()
def deeper(self, req, resp):
    assert isinstance(req, webob.BaseRequest)
    assert isinstance(resp, webob.Response)
    return '/deeper'