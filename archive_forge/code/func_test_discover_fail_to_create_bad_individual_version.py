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
def test_discover_fail_to_create_bad_individual_version(self):
    self.requests_mock.get(V2_URL, text=V2_VERSION_ENTRY)
    self.requests_mock.get(V3_URL, text=V3_VERSION_ENTRY)
    self.assertVersionNotAvailable(auth_url=V2_URL, version=3)
    self.assertVersionNotAvailable(auth_url=V3_URL, version=2)