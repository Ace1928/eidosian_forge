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
def test_app_wrap(self):

    class RootController(object):
        pass
    wrapped_apps = []

    def wrap(app):
        wrapped_apps.append(app)
        return app
    make_app(RootController(), wrap_app=wrap, debug=True)
    assert len(wrapped_apps) == 1