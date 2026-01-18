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
def test_http_manual_redirect_delete(self, mock_request):
    mock_request.side_effect = [fakes.FakeHTTPResponse(302, 'Found', {'location': 'http://example.com:8004/foo/bar'}, ''), fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/json'}, 'invalid-json')]
    client = http.HTTPClient('http://example.com:8004/foo')
    resp, body = client.json_request('DELETE', '')
    mock_request.assert_has_calls([mock.call('DELETE', 'http://example.com:8004/foo', allow_redirects=False, headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-heatclient'}), mock.call('DELETE', 'http://example.com:8004/foo/bar', allow_redirects=False, headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-heatclient'})])