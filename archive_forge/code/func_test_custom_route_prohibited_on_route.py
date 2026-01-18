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
def test_custom_route_prohibited_on_route(self):
    try:

        class RootController(object):

            @expose(route='some-path')
            def _route(self):
                return 'Hello, World!'
    except ValueError:
        pass
    else:
        raise AssertionError('_route cannot be used with a custom path segment')