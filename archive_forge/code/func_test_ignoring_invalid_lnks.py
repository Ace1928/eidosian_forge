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
def test_ignoring_invalid_lnks(self):
    version_list = [{'id': 'v3.0', 'links': [{'href': V3_URL, 'rel': 'self'}], 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'id': 'v3.1', 'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED}, {'media-types': V3_MEDIA_TYPES, 'status': 'stable', 'updated': UPDATED, 'links': [{'href': V3_URL, 'rel': 'self'}]}]
    text = jsonutils.dumps({'versions': version_list})
    self.requests_mock.get(BASE_URL, text=text)
    with self.deprecations.expect_deprecations_here():
        disc = discover.Discover(auth_url=BASE_URL)
    versions = disc.raw_version_data()
    self.assertEqual(3, len(versions))
    versions = disc.version_data()
    self.assertEqual(1, len(versions))