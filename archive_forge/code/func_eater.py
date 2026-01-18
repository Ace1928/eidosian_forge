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
def eater(self, req, resp, id, dummy=None, *args, **kwargs):
    data = ['%s=%s' % (key, kwargs[key]) for key in sorted(kwargs.keys())]
    return 'eater: %s, %s, %s' % (id, dummy, ', '.join(list(args) + data))