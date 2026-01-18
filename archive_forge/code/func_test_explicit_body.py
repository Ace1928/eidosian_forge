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
def test_explicit_body(self):
    r = self.app_.get('/explicit_body/')
    self.assertEqual(r.status_int, 200)
    self.assertEqual(r.body, b'Hello, World!')