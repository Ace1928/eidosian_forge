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
def test_lesser_version_than_required(self):
    versions = fixture.DiscoveryList(BASE_URL, v3_id='v3.4')
    self.requests_mock.get(BASE_URL, json=versions)
    self.assertVersionNotAvailable(auth_url=BASE_URL, version=(3, 6))