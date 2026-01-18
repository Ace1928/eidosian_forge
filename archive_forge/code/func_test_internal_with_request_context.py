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
def test_internal_with_request_context(self):
    r = self.app_.get('/internal_with_context')
    assert r.status_int == 200
    assert json.loads(r.body.decode()) == {'foo': 'bar'}