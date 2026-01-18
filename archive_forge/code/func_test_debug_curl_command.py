import socket
from unittest import mock
import io
from keystoneauth1 import adapter
from oslo_serialization import jsonutils
import testtools
from heatclient.common import http
from heatclient.common import utils
from heatclient import exc
from heatclient.tests.unit import fakes
def test_debug_curl_command(self, mock_request):
    with mock.patch('logging.Logger.debug') as mock_logging_debug:
        ssl_connection_params = {'ca_file': 'TEST_CA', 'cert_file': 'TEST_CERT', 'key_file': 'TEST_KEY', 'insecure': 'TEST_NSA'}
        headers = {'key': 'value'}
        mock_logging_debug.return_value = None
        client = http.HTTPClient('http://foo')
        client.ssl_connection_params = ssl_connection_params
        client.log_curl_request('GET', '/bar', {'headers': headers, 'data': 'text'})
        mock_logging_debug.assert_called_with("curl -g -i -X GET -H 'key: value' --key TEST_KEY --cert TEST_CERT --cacert TEST_CA -k -d 'text' http://foo/bar")