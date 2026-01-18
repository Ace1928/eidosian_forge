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
def test_endpoint_data_token_endpoint_no_discover(self):
    plugin = token_endpoint.Token(endpoint=V3_URL, token='bogus')
    data = plugin.get_endpoint_data(self.session, discover_versions=False)
    self.assertEqual(data.api_version, (3, 0))
    self.assertEqual(data.url, V3_URL)
    self.assertEqual(plugin.get_api_major_version(self.session), (3, 0))
    self.assertEqual(plugin.get_endpoint(self.session), V3_URL)