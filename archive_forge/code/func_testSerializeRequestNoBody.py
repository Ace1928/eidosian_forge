import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testSerializeRequestNoBody(self):
    request = http_wrapper.Request(body=None, headers={'content-type': 'protocol/version'})
    expected_serialized_request = '\n'.join(['GET  HTTP/1.1', 'Content-Type: protocol/version', 'MIME-Version: 1.0', 'Host: ', '', ''])
    batch_request = batch.BatchHttpRequest('https://www.example.com')
    self.assertEqual(expected_serialized_request, batch_request._SerializeRequest(request))