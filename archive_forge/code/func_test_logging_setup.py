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
def test_logging_setup(self):

    class RootController(object):

        @expose()
        def index(self):
            import logging
            logging.getLogger('pecantesting').info('HELLO WORLD')
            return 'HELLO WORLD'
    f = StringIO()
    app = TestApp(make_app(RootController(), logging={'loggers': {'pecantesting': {'level': 'INFO', 'handlers': ['memory']}}, 'handlers': {'memory': {'level': 'INFO', 'class': 'logging.StreamHandler', 'stream': f}}}))
    app.get('/')
    assert f.getvalue() == 'HELLO WORLD\n'