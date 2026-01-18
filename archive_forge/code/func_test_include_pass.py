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
def test_include_pass(self, mock_request):
    fake200 = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/octet-stream'}, '')
    mock_request.return_value = fake200
    client = http.HTTPClient('http://example.com:8004')
    resp = client.raw_request('GET', '')
    self.assertEqual(200, resp.status_code)
    client.username = 'user'
    client.password = 'pass'
    client.include_pass = True
    resp = client.raw_request('GET', '')
    self.assertEqual(200, resp.status_code)
    client.auth_token = 'abcd1234'
    resp = client.raw_request('GET', '')
    self.assertEqual(200, resp.status_code)
    mock_request.assert_has_calls([mock.call('GET', 'http://example.com:8004', allow_redirects=False, headers={'Content-Type': 'application/octet-stream', 'User-Agent': 'python-heatclient'}), mock.call('GET', 'http://example.com:8004', allow_redirects=False, headers={'Content-Type': 'application/octet-stream', 'User-Agent': 'python-heatclient', 'X-Auth-Key': 'pass', 'X-Auth-User': 'user'}), mock.call('GET', 'http://example.com:8004', allow_redirects=False, headers={'Content-Type': 'application/octet-stream', 'User-Agent': 'python-heatclient', 'X-Auth-Token': 'abcd1234', 'X-Auth-Key': 'pass', 'X-Auth-User': 'user'})])