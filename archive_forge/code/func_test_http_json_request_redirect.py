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
def test_http_json_request_redirect(self, mock_request):
    mock_request.side_effect = [fakes.FakeHTTPResponse(302, 'Found', {'location': 'http://example.com:8004'}, ''), fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/json'}, '{}')]
    client = http.HTTPClient('http://example.com:8004')
    resp, body = client.json_request('GET', '')
    self.assertEqual(200, resp.status_code)
    self.assertEqual({}, body)
    mock_request.assert_has_calls([mock.call('GET', 'http://example.com:8004', allow_redirects=False, headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-heatclient'}), mock.call('GET', 'http://example.com:8004', allow_redirects=False, headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-heatclient'})])