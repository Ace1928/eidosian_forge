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
def test_keyword_argument_with_encoded_plus(self):
    r = self.app_.get('/?id=foo%2Bbar')
    assert r.status_int == 200
    assert r.body == b'index: foo+bar'