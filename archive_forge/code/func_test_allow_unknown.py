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
def test_allow_unknown(self):
    status = 'abcdef'
    version_list = fixture.DiscoveryList(BASE_URL, v2=False, v3_status=status)
    self.requests_mock.get(BASE_URL, json=version_list)
    disc = discover.Discover(self.session, BASE_URL)
    versions = disc.version_data()
    self.assertEqual(0, len(versions))
    versions = disc.version_data(allow_unknown=True)
    self.assertEqual(1, len(versions))
    self.assertEqual(status, versions[0]['raw_status'])
    self.assertEqual(V3_URL, versions[0]['url'])
    self.assertEqual((3, 0), versions[0]['version'])