from http import client as http_client
import io
from unittest import mock
from oslo_serialization import jsonutils
import socket
from magnumclient.common.apiclient.exceptions import GatewayTimeout
from magnumclient.common.apiclient.exceptions import MultipleChoices
from magnumclient.common import httpclient as http
from magnumclient import exceptions as exc
from magnumclient.tests import utils
def test_401_unauthorized_exception(self):
    error_body = _get_error_body(err_type=ERROR_LIST_WITH_DETAIL)
    fake_resp = utils.FakeResponse({'content-type': 'text/plain'}, io.StringIO(error_body), version=1, status=401)
    client = http.HTTPClient('http://localhost/')
    client.get_connection = lambda *a, **kw: utils.FakeConnection(fake_resp)
    self.assertRaises(exc.Unauthorized, client.json_request, 'GET', '/v1/resources')