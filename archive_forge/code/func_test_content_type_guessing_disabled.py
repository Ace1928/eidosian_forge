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
def test_content_type_guessing_disabled(self):

    class ResourceController(object):

        def __init__(self, name):
            self.name = name
            assert self.name == 'file.html'

        @expose('json')
        def index(self):
            return dict(name=self.name)

    class RootController(object):

        @expose()
        def _lookup(self, name, *remainder):
            return (ResourceController(name), remainder)
    app = TestApp(Pecan(RootController(), guess_content_type_from_ext=False))
    r = app.get('/file.html/')
    assert r.status_int == 200
    result = dict(json.loads(r.body.decode()))
    assert result == {'name': 'file.html'}
    r = app.get('/file.html')
    assert r.status_int == 302
    r = r.follow()
    result = dict(json.loads(r.body.decode()))
    assert result == {'name': 'file.html'}