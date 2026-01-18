import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testNewId(self):
    batch_request = batch.BatchHttpRequest('https://www.example.com')
    for i in range(100):
        self.assertEqual(str(i), batch_request._NewId())