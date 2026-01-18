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
@mock.patch.object(jsonutils, 'dumps')
def test_kwargs_with_files(self, mock_dumps):
    fake = fakes.FakeHTTPResponse(200, 'OK', {'content-type': 'application/json'}, {})
    mock_dumps.return_value = "{'files': test}}"
    data = io.BytesIO(b'test')
    kwargs = {'endpoint_override': 'http://no.where/', 'data': {'files': data}}
    client = http.SessionClient(mock.ANY)
    self.request.return_value = (fake, {})
    resp = client.request('', 'GET', **kwargs)
    self.assertEqual({'endpoint_override': 'http://no.where/', 'data': "{'files': test}}", 'headers': {'Content-Type': 'application/json'}, 'user_agent': 'python-heatclient', 'raise_exc': False}, self.request.call_args[1])
    self.assertEqual(200, resp.status_code)