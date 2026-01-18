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
def test_custom_objects(self):

    class CustomRequest(Request):

        @property
        def headers(self):
            headers = super(CustomRequest, self).headers
            headers['X-Custom-Request'] = 'ABC'
            return headers

    class CustomResponse(Response):

        @property
        def headers(self):
            headers = super(CustomResponse, self).headers
            headers['X-Custom-Response'] = 'XYZ'
            return headers

    class RootController(object):

        @expose()
        def index(self):
            return request.headers.get('X-Custom-Request')
    app = TestApp(Pecan(RootController(), request_cls=CustomRequest, response_cls=CustomResponse))
    r = app.get('/')
    assert r.body == b'ABC'
    assert r.headers.get('X-Custom-Response') == 'XYZ'