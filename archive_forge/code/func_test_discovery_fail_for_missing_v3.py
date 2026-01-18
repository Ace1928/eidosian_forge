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
def test_discovery_fail_for_missing_v3(self):
    versions = fixture.DiscoveryList(v2=True, v3=False)
    self.requests_mock.get(BASE_URL, status_code=300, json=versions)
    with self.deprecations.expect_deprecations_here():
        disc = discover.Discover(auth_url=BASE_URL)
    self.assertRaises(exceptions.DiscoveryFailure, disc.create_client, version=(3, 0))