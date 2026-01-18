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
def test_http_json_request_non_json_resp_cont_type(self, mock_request):
    mock_request.return_value = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'not/json'}, '{}')
    client = http.HTTPClient('http://example.com:8004')
    resp, body = client.json_request('GET', '', body='test-body')
    self.assertEqual(200, resp.status_code)
    self.assertIsNone(body)
    mock_request.assert_called_with('GET', 'http://example.com:8004', body='test-body', allow_redirects=False, headers={'Content-Type': 'application/json', 'Accept': 'application/json', 'User-Agent': 'python-heatclient'})