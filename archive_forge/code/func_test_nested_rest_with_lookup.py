from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_nested_rest_with_lookup(self):

    class SubController(RestController):

        @expose()
        def get_all(self):
            return 'SUB'

    class FinalController(RestController):

        def __init__(self, id_):
            self.id_ = id_

        @expose()
        def get_all(self):
            return 'FINAL-%s' % self.id_

        @expose()
        def post(self):
            return 'POST-%s' % self.id_

    class LookupController(RestController):
        sub = SubController()

        def __init__(self, id_):
            self.id_ = id_

        @expose()
        def _lookup(self, id_, *remainder):
            return (FinalController(id_), remainder)

        @expose()
        def get_all(self):
            raise AssertionError('Never Reached')

        @expose()
        def post(self):
            return 'POST-LOOKUP-%s' % self.id_

        @expose()
        def put(self, id_):
            return 'PUT-LOOKUP-%s-%s' % (self.id_, id_)

        @expose()
        def delete(self, id_):
            return 'DELETE-LOOKUP-%s-%s' % (self.id_, id_)

    class FooController(RestController):

        @expose()
        def _lookup(self, id_, *remainder):
            return (LookupController(id_), remainder)

        @expose()
        def get_one(self, id_):
            return 'GET ONE'

        @expose()
        def get_all(self):
            return 'INDEX'

        @expose()
        def post(self):
            return 'POST'

        @expose()
        def put(self, id_):
            return 'PUT-%s' % id_

        @expose()
        def delete(self, id_):
            return 'DELETE-%s' % id_

    class RootController(RestController):
        foo = FooController()
    app = TestApp(make_app(RootController()))
    r = app.get('/foo')
    assert r.status_int == 200
    assert r.body == b'INDEX'
    r = app.post('/foo')
    assert r.status_int == 200
    assert r.body == b'POST'
    r = app.get('/foo/1')
    assert r.status_int == 200
    assert r.body == b'GET ONE'
    r = app.post('/foo/1')
    assert r.status_int == 200
    assert r.body == b'POST-LOOKUP-1'
    r = app.put('/foo/1')
    assert r.status_int == 200
    assert r.body == b'PUT-1'
    r = app.delete('/foo/1')
    assert r.status_int == 200
    assert r.body == b'DELETE-1'
    r = app.put('/foo/1/2')
    assert r.status_int == 200
    assert r.body == b'PUT-LOOKUP-1-2'
    r = app.delete('/foo/1/2')
    assert r.status_int == 200
    assert r.body == b'DELETE-LOOKUP-1-2'
    r = app.get('/foo/1/2')
    assert r.status_int == 200
    assert r.body == b'FINAL-2'
    r = app.post('/foo/1/2')
    assert r.status_int == 200
    assert r.body == b'POST-2'