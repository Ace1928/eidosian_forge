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
def test_abort_keeps_traceback(self):
    last_exc, last_traceback = (None, None)
    try:
        try:
            raise Exception('Bottom Exception')
        except:
            abort(404)
    except Exception:
        last_exc, _, last_traceback = sys.exc_info()
    assert last_exc is HTTPNotFound
    assert 'Bottom Exception' in traceback.format_tb(last_traceback)[-1]