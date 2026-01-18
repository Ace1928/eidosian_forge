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
def test_ignore_entry_without_status(self):
    v3 = V3_VERSION.copy()
    del v3['status']
    self.requests_mock.get(BASE_URL, status_code=300, text=_create_version_list([v3, V2_VERSION]))
    self.assertCreatesV2(auth_url=BASE_URL)