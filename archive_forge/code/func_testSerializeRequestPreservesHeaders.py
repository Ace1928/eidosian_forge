import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testSerializeRequestPreservesHeaders(self):
    request = http_wrapper.Request(body='Hello World', headers={'content-type': 'protocol/version', 'key': 'value'})
    batch_request = batch.BatchHttpRequest('https://www.example.com')
    self.assertTrue('key: value\n' in batch_request._SerializeRequest(request))