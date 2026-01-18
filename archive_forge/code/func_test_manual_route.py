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
def test_manual_route(self):

    class SubController(object):

        @expose(route='some-path')
        def some_path(self):
            return 'Hello, World!'

    class RootController(object):
        pass
    route(RootController, 'some-controller', SubController())
    app = TestApp(Pecan(RootController()))
    r = app.get('/some-controller/some-path/')
    assert r.status_int == 200
    assert r.body == b'Hello, World!'
    r = app.get('/some-controller/some_path/', expect_errors=True)
    assert r.status_int == 404