import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testNoAttempts(self):
    desired_url = 'https://www.example.com'
    batch_api_request = batch.BatchApiRequest(batch_url=desired_url)
    batch_api_request.Add(FakeService(), 'unused', None, {'desired_request': http_wrapper.Request(desired_url, 'POST', {'content-type': 'multipart/mixed; boundary="None"', 'content-length': 80}, 'x' * 80)})
    api_request_responses = batch_api_request.Execute(None, max_retries=0)
    self.assertEqual(1, len(api_request_responses))
    self.assertIsNone(api_request_responses[0].response)
    self.assertIsNone(api_request_responses[0].exception)