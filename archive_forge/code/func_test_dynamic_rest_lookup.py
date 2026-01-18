from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_dynamic_rest_lookup(self):

    class BarController(RestController):

        @expose()
        def get_all(self):
            return 'BAR'

        @expose()
        def put(self):
            return 'PUT_BAR'

        @expose()
        def delete(self):
            return 'DELETE_BAR'

    class BarsController(RestController):

        @expose()
        def _lookup(self, id_, *remainder):
            return (BarController(), remainder)

        @expose()
        def get_all(self):
            return 'BARS'

        @expose()
        def post(self):
            return 'POST_BARS'

    class FooController(RestController):
        bars = BarsController()

        @expose()
        def get_all(self):
            return 'FOO'

        @expose()
        def put(self):
            return 'PUT_FOO'

        @expose()
        def delete(self):
            return 'DELETE_FOO'

    class FoosController(RestController):

        @expose()
        def _lookup(self, id_, *remainder):
            return (FooController(), remainder)

        @expose()
        def get_all(self):
            return 'FOOS'

        @expose()
        def post(self):
            return 'POST_FOOS'

    class RootController(RestController):
        foos = FoosController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foos')
    assert r.status_int == 200
    assert r.body == b'FOOS'
    r = app.post('/foos')
    assert r.status_int == 200
    assert r.body == b'POST_FOOS'
    r = app.get('/foos/foo')
    assert r.status_int == 200
    assert r.body == b'FOO'
    r = app.put('/foos/foo')
    assert r.status_int == 200
    assert r.body == b'PUT_FOO'
    r = app.delete('/foos/foo')
    assert r.status_int == 200
    assert r.body == b'DELETE_FOO'
    r = app.get('/foos/foo/bars')
    assert r.status_int == 200
    assert r.body == b'BARS'
    r = app.post('/foos/foo/bars')
    assert r.status_int == 200
    assert r.body == b'POST_BARS'
    r = app.get('/foos/foo/bars/bar')
    assert r.status_int == 200
    assert r.body == b'BAR'
    r = app.put('/foos/foo/bars/bar')
    assert r.status_int == 200
    assert r.body == b'PUT_BAR'
    r = app.delete('/foos/foo/bars/bar')
    assert r.status_int == 200
    assert r.body == b'DELETE_BAR'