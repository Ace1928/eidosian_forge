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
def test_manual_route_conflict(self):

    class SubController(object):
        pass

    class RootController(object):

        @expose()
        def hello(self):
            return 'Hello, World!'
    self.assertRaises(RuntimeError, route, RootController, 'hello', SubController())