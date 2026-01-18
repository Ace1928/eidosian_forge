import json
import re
from unittest import mock
from testtools import matchers
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import http_basic
from keystoneauth1 import noauth
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
from keystoneauth1 import token_endpoint
def test_endpoint_data_token_endpoint_adapter(self):
    mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
    plugin = token_endpoint.Token(endpoint=V3_URL, token='bogus')
    client = adapter.Adapter(session.Session(plugin))
    data = client.get_endpoint_data()
    self.assertEqual(data.api_version, (3, 0))
    self.assertEqual(data.url, V3_URL)
    self.assertEqual(client.get_api_major_version(), (3, 0))
    self.assertEqual(client.get_endpoint(), V3_URL)
    self.assertTrue(mock.called_once)