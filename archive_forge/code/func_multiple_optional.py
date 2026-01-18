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
def multiple_optional(self, req, resp, one=None, two=None, three=None):
    return 'multiple_optional: %s, %s, %s' % (one, two, three)