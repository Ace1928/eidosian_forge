import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testInternalExecuteWithNonMultipartResponse(self):
    with mock.patch.object(http_wrapper, 'MakeRequest', autospec=True) as mock_request:
        self.__ConfigureMock(mock_request, http_wrapper.Request('https://www.example.com', 'POST', {'content-type': 'multipart/mixed; boundary="None"', 'content-length': 80}, 'x' * 80), http_wrapper.Response({'status': '200', 'content-type': 'blah/blah'}, '', None))
        batch_request = batch.BatchHttpRequest('https://www.example.com')
        self.assertRaises(exceptions.BatchError, batch_request._Execute, None)