from json import dumps, loads
import unittest
import struct
import sys
import warnings
from webtest import TestApp
from pecan import abort, expose, make_app, response, redirect
from pecan.rest import RestController
from pecan.tests import PecanTestCase
def test_alternate_route(self):

    class RootController(RestController):

        @expose(route='some-path')
        def get_all(self):
            return 'Hello, World!'
    self.assertRaises(ValueError, RootController)