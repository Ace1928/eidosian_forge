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
def test_get_connection_params_with_ssl_params(self):
    endpoint = 'https://magnum-host:6385'
    ssl_args = {'ca_file': '/path/to/ca_file', 'cert_file': '/path/to/cert_file', 'key_file': '/path/to/key_file', 'insecure': True}
    expected_kwargs = {'timeout': DEFAULT_TIMEOUT}
    expected_kwargs.update(ssl_args)
    expected = (HTTPS_CLASS, ('magnum-host', 6385, ''), expected_kwargs)
    params = http.HTTPClient.get_connection_params(endpoint, **ssl_args)
    self.assertEqual(expected, params)