from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_bad_rest(self):

    class ThingsController(RestController):
        pass

    class RootController(object):
        things = ThingsController()
    app = TestApp(make_app(RootController()))
    r = app.get('/things', status=405)
    assert r.status_int == 405
    r = app.get('/things/1', status=405)
    assert r.status_int == 405
    r = app.post('/things', {'value': 'one'}, status=405)
    assert r.status_int == 405
    r = app.get('/things/1/edit', status=405)
    assert r.status_int == 405
    r = app.put('/things/1', {'value': 'ONE'}, status=405)
    r = app.get('/things/1?_method=put', {'value': 'ONE!'}, status=405)
    assert r.status_int == 405
    r = app.post('/things/1?_method=put', {'value': 'ONE!'}, status=405)
    assert r.status_int == 405
    r = app.get('/things/1/delete', status=405)
    assert r.status_int == 405
    r = app.delete('/things/1', status=405)
    assert r.status_int == 405
    r = app.get('/things/1?_method=DELETE', status=405)
    assert r.status_int == 405
    r = app.post('/things/1?_method=DELETE', status=405)
    assert r.status_int == 405
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        r = app.request('/things', method='TRACE', status=405)
        assert r.status_int == 405