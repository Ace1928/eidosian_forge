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
def test_http_request_socket_timeout(self, mock_request):
    headers = {'User-Agent': 'python-heatclient'}
    mock_request.side_effect = [socket.timeout]
    client = http.HTTPClient('http://example.com:8004')
    self.assertRaises(exc.CommunicationError, client._http_request, '/', 'GET')
    mock_request.assert_called_with('GET', 'http://example.com:8004/', allow_redirects=False, headers=headers)