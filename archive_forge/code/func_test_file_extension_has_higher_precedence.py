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
def test_file_extension_has_higher_precedence(self):
    r = self.app_.get('/index.html', headers={'Accept': 'application/json,text/html;q=0.9,*/*;q=0.8'})
    assert r.status_int == 200
    assert r.content_type == 'text/html'