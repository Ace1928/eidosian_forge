from http import client as http_client
from io import StringIO
from unittest import mock
from oslo_serialization import jsonutils
from zunclient import api_versions
from zunclient.common.apiclient import exceptions
from zunclient.common import httpclient as http
from zunclient import exceptions as exc
from zunclient.tests.unit import utils
def test_get_connection_params_with_trailing_slash(self):
    endpoint = 'http://zun-host:6385/'
    expected = (HTTP_CLASS, ('zun-host', 6385, ''), {'timeout': DEFAULT_TIMEOUT})
    params = http.HTTPClient.get_connection_params(endpoint)
    self.assertEqual(expected, params)