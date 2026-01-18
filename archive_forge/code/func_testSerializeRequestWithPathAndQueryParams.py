import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testSerializeRequestWithPathAndQueryParams(self):
    request = http_wrapper.Request(url='my/path?query=param', body='Hello World', headers={'content-type': 'protocol/version'})
    expected_serialized_request = '\n'.join(['GET my/path?query=param HTTP/1.1', 'Content-Type: protocol/version', 'MIME-Version: 1.0', 'content-length: 11', 'Host: ', '', 'Hello World'])
    batch_request = batch.BatchHttpRequest('https://www.example.com')
    self.assertEqual(expected_serialized_request, batch_request._SerializeRequest(request))