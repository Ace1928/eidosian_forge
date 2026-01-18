from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_custom_delete(self):

    class OthersController(object):

        @expose()
        def index(self):
            return 'DELETE'

        @expose()
        def reset(self, id):
            return str(id)

    class ThingsController(RestController):
        others = OthersController()

        @expose()
        def delete_fail(self):
            abort(500)

    class RootController(object):
        things = ThingsController()
    app = TestApp(make_app(RootController()))
    r = app.delete('/things/delete_fail', status=405)
    assert r.status_int == 405
    r = app.get('/things/delete_fail?_method=delete', status=405)
    assert r.status_int == 405
    r = app.post('/things/delete_fail', {'_method': 'delete'}, status=405)
    assert r.status_int == 405
    r = app.delete('/things/others/')
    assert r.status_int == 200
    assert r.body == b'DELETE'
    r = app.get('/things/others/?_method=delete', status=405)
    assert r.status_int == 405
    r = app.post('/things/others/', {'_method': 'delete'})
    assert r.status_int == 200
    assert r.body == b'DELETE'
    r = app.delete('/things/others/reset/1')
    assert r.status_int == 200
    assert r.body == b'1'
    r = app.get('/things/others/reset/1?_method=delete', status=405)
    assert r.status_int == 405
    r = app.post('/things/others/reset/1', {'_method': 'delete'})
    assert r.status_int == 200
    assert r.body == b'1'