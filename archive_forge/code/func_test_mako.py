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
@unittest.skipIf('mako' not in builtin_renderers, 'Mako not installed')
def test_mako(self):

    class RootController(object):

        @expose('mako:mako.html')
        def index(self, name='Jonathan'):
            return dict(name=name)

        @expose('mako:mako_bad.html')
        def badtemplate(self):
            return dict()
    app = TestApp(Pecan(RootController(), template_path=self.template_path))
    r = app.get('/')
    assert r.status_int == 200
    assert b'<h1>Hello, Jonathan!</h1>' in r.body
    r = app.get('/index.html?name=World')
    assert r.status_int == 200
    assert b'<h1>Hello, World!</h1>' in r.body
    error_msg = None
    try:
        r = app.get('/badtemplate.html')
    except Exception as e:
        for error_f in error_formatters:
            error_msg = error_f(e)
            if error_msg:
                break
    assert error_msg is not None