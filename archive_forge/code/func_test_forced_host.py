from tests.compat import mock, unittest
from boto.compat import http_client
from boto.sns import connect_to_region
def test_forced_host(self):
    https = http_client.HTTPConnection
    mpo = mock.patch.object
    with mpo(https, 'request') as mock_request:
        with mpo(https, 'getresponse', return_value=StubResponse()):
            with self.assertRaises(self.connection.ResponseError):
                self.connection.list_platform_applications()
    call = mock_request.call_args_list[0]
    headers = call[0][3]
    self.assertTrue('Host' in headers)
    self.assertEqual(headers['Host'], 'sns.us-west-2.amazonaws.com')