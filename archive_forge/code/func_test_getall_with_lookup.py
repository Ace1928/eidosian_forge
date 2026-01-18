from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_getall_with_lookup(self):

    class LookupController(RestController):

        def __init__(self, _id):
            self._id = _id

        @expose()
        def get_all(self):
            return 'ID: %s' % self._id

    class ThingsController(RestController):
        data = ['zero', 'one', 'two', 'three']

        @expose()
        def _lookup(self, _id, *remainder):
            return (LookupController(_id), remainder)

        @expose('json')
        def get_all(self):
            return dict(items=self.data)

    class RootController(object):
        things = ThingsController()
    app = TestApp(make_app(RootController()))
    for path in ('/things', '/things/'):
        r = app.get(path)
        assert r.status_int == 200
        assert r.body == dumps(dict(items=ThingsController.data)).encode('utf-8')
    r = app.get('/things/foo')
    assert r.status_int == 200
    assert r.body == b'ID: foo'