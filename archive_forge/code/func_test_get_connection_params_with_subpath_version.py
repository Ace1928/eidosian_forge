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
def test_get_connection_params_with_subpath_version(self):
    endpoint = 'http://magnum-host:6385/magnum/v1'
    expected = (HTTP_CLASS, ('magnum-host', 6385, '/magnum'), {'timeout': DEFAULT_TIMEOUT})
    params = http.HTTPClient.get_connection_params(endpoint)
    self.assertEqual(expected, params)