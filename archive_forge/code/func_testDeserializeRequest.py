import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testDeserializeRequest(self):
    serialized_payload = '\n'.join(['GET  HTTP/1.1', 'Content-Type: protocol/version', 'MIME-Version: 1.0', 'content-length: 11', 'key: value', 'Host: ', '', 'Hello World'])
    example_url = 'https://www.example.com'
    expected_response = http_wrapper.Response({'content-length': str(len('Hello World')), 'Content-Type': 'protocol/version', 'key': 'value', 'MIME-Version': '1.0', 'status': '', 'Host': ''}, 'Hello World', example_url)
    batch_request = batch.BatchHttpRequest(example_url)
    self.assertEqual(expected_response, batch_request._DeserializeResponse(serialized_payload))