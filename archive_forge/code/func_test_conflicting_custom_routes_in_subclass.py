import sys
import os
import json
import traceback
import warnings
from io import StringIO, BytesIO
import webob
from webob.exc import HTTPNotFound
from webtest import TestApp
from pecan import (
from pecan.templating import (
from pecan.decorators import accept_noncanonical
from pecan.tests import PecanTestCase
import unittest
def test_conflicting_custom_routes_in_subclass(self):

    class BaseController(object):

        @expose(route='testing')
        def foo(self):
            return request.path

    class ChildController(BaseController):
        pass

    class RootController(BaseController):
        child = ChildController()
    app = TestApp(Pecan(RootController()))
    r = app.get('/testing/')
    assert r.body == b'/testing/'
    r = app.get('/child/testing/')
    assert r.body == b'/child/testing/'