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
def test_data_for_url(self):
    mock = self.requests_mock.get(V3_URL, status_code=200, json=V3_VERSION_ENTRY)
    disc = discover.Discover(self.session, V3_URL)
    for url in (V3_URL, V3_URL + '/'):
        data = disc.versioned_data_for(url=url)
        self.assertEqual(data['version'], (3, 0))
        self.assertEqual(data['raw_status'], 'stable')
        self.assertEqual(data['url'], V3_URL)
    self.assertTrue(mock.called_once)