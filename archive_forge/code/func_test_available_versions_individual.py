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
def test_available_versions_individual(self):
    self.requests_mock.get(V3_URL, status_code=200, text=V3_VERSION_ENTRY)
    versions = discover.available_versions(V3_URL)
    for v in versions:
        self.assertEqual(v['id'], 'v3.0')
        self.assertEqual(v['status'], 'stable')
        self.assertIn('media-types', v)
        self.assertIn('links', v)