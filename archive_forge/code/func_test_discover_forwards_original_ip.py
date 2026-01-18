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
def test_discover_forwards_original_ip(self):
    self.requests_mock.get(BASE_URL, status_code=300, text=V3_VERSION_LIST)
    ip = '192.168.1.1'
    self.assertCreatesV3(auth_url=BASE_URL, original_ip=ip)
    self.assertThat(self.requests_mock.last_request.headers['forwarded'], matchers.Contains(ip))