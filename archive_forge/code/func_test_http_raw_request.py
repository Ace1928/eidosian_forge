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
def test_http_raw_request(self, mock_request):
    headers = {'Content-Type': 'application/octet-stream', 'User-Agent': 'python-heatclient'}
    mock_request.return_value = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/octet-stream'}, '')
    client = http.HTTPClient('http://example.com:8004')
    resp = client.raw_request('GET', '')
    self.assertEqual(200, resp.status_code)
    self.assertEqual('', ''.join([x for x in resp.content]))
    mock_request.assert_called_with('GET', 'http://example.com:8004', allow_redirects=False, headers=headers)