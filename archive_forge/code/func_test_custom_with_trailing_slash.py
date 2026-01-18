from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_custom_with_trailing_slash(self):

    class CustomController(RestController):
        _custom_actions = {'detail': ['GET'], 'create': ['POST'], 'update': ['PUT'], 'remove': ['DELETE']}

        @expose()
        def detail(self):
            return 'DETAIL'

        @expose()
        def create(self):
            return 'CREATE'

        @expose()
        def update(self, id):
            return id

        @expose()
        def remove(self, id):
            return id
    app = TestApp(make_app(CustomController()))
    r = app.get('/detail')
    assert r.status_int == 200
    assert r.body == b'DETAIL'
    r = app.get('/detail/')
    assert r.status_int == 200
    assert r.body == b'DETAIL'
    r = app.post('/create')
    assert r.status_int == 200
    assert r.body == b'CREATE'
    r = app.post('/create/')
    assert r.status_int == 200
    assert r.body == b'CREATE'
    r = app.put('/update/123')
    assert r.status_int == 200
    assert r.body == b'123'
    r = app.put('/update/123/')
    assert r.status_int == 200
    assert r.body == b'123'
    r = app.delete('/remove/456')
    assert r.status_int == 200
    assert r.body == b'456'
    r = app.delete('/remove/456/')
    assert r.status_int == 200
    assert r.body == b'456'