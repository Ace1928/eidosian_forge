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
def test_custom_route_prohibited_on_generic_controllers(self):
    try:

        class RootController(object):

            @expose(generic=True)
            def foo(self):
                return 'Hello, World!'

            @foo.when(method='POST', route='some-path')
            def handle_post(self):
                return 'POST!'
    except ValueError:
        pass
    else:
        raise AssertionError('generic controllers cannot be used with a custom path segment')