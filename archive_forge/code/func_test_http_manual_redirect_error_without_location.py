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
def test_http_manual_redirect_error_without_location(self, mock_request):
    mock_request.return_value = fakes.FakeHTTPResponse(302, 'Found', {}, '')
    client = http.HTTPClient('http://example.com:8004/foo')
    self.assertRaises(exc.InvalidEndpoint, client.json_request, 'DELETE', '')
    mock_request.assert_called_once_with('DELETE', 'http://example.com:8004/foo', allow_redirects=False, headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-heatclient'})