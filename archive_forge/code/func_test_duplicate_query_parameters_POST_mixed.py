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
def test_duplicate_query_parameters_POST_mixed(self):
    r = self.app_.post('/variable_kwargs?list=1&list=2', {'list': ['3', '4']})
    assert r.status_int == 200
    assert r.body == b"variable_kwargs: list=['1', '2', '3', '4']"