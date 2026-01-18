import re
import uuid
from keystoneauth1 import fixture
from oslo_serialization import jsonutils
from testtools import matchers
from keystoneclient import _discover
from keystoneclient.auth import token_endpoint
from keystoneclient import client
from keystoneclient import discover
from keystoneclient import exceptions
from keystoneclient import session
from keystoneclient.tests.unit import utils
from keystoneclient.v2_0 import client as v2_client
from keystoneclient.v3 import client as v3_client
def test_available_keystone_data(self):
    self.requests_mock.get(BASE_URL, status_code=300, text=V3_VERSION_LIST)
    with self.deprecations.expect_deprecations_here():
        disc = discover.Discover(auth_url=BASE_URL)
    versions = disc.version_data()
    self.assertEqual((2, 0), versions[0]['version'])
    self.assertEqual('stable', versions[0]['raw_status'])
    self.assertEqual(V2_URL, versions[0]['url'])
    self.assertEqual((3, 0), versions[1]['version'])
    self.assertEqual('stable', versions[1]['raw_status'])
    self.assertEqual(V3_URL, versions[1]['url'])
    version = disc.data_for('v3.0')
    self.assertEqual((3, 0), version['version'])
    self.assertEqual('stable', version['raw_status'])
    self.assertEqual(V3_URL, version['url'])
    version = disc.data_for(2)
    self.assertEqual((2, 0), version['version'])
    self.assertEqual('stable', version['raw_status'])
    self.assertEqual(V2_URL, version['url'])
    self.assertIsNone(disc.url_for('v4'))
    self.assertEqual(V3_URL, disc.url_for('v3'))
    self.assertEqual(V2_URL, disc.url_for('v2'))