import textwrap
import unittest
import mock
from six.moves import http_client
from six.moves import range  # pylint:disable=redefined-builtin
from six.moves.urllib import parse
from apitools.base.py import batch
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
def testPublicExecute(self):

    def LocalCallback(response, exception):
        self.assertEqual({'status': '418'}, response.info)
        self.assertEqual('Teapot', response.content)
        self.assertIsNone(response.request_url)
        self.assertIsInstance(exception, exceptions.HttpError)
    global_callback = mock.Mock()
    batch_request = batch.BatchHttpRequest('https://www.example.com', global_callback)
    with mock.patch.object(batch.BatchHttpRequest, '_Execute', autospec=True) as mock_execute:
        mock_execute.return_value = None
        test_requests = {'0': batch.RequestResponseAndHandler(None, http_wrapper.Response({'status': '200'}, 'Hello!', None), None), '1': batch.RequestResponseAndHandler(None, http_wrapper.Response({'status': '418'}, 'Teapot', None), LocalCallback)}
        batch_request._BatchHttpRequest__request_response_handlers = test_requests
        batch_request.Execute(None)
        self.assertEqual(len(test_requests), global_callback.call_count)