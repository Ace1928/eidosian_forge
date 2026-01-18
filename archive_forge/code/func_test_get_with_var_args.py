from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_get_with_var_args(self):

    class OthersController(object):

        @expose()
        def index(self, one, two, three):
            return 'NESTED: %s, %s, %s' % (one, two, three)

    class ThingsController(RestController):
        others = OthersController()

        @expose()
        def get_one(self, *args):
            return ', '.join(args)

    class RootController(object):
        things = ThingsController()
    app = TestApp(make_app(RootController()))
    r = app.get('/things/one/two/three')
    assert r.status_int == 200
    assert r.body == b'one, two, three'
    r = app.get('/things/one/two/three/others/')
    assert r.status_int == 200
    assert r.body == b'NESTED: one, two, three'