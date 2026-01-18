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
def assertCreatesV2(self, **kwargs):
    self.requests_mock.post('%s/tokens' % V2_URL, text=V2_AUTH_RESPONSE)
    kwargs.setdefault('username', 'foo')
    kwargs.setdefault('password', 'bar')
    keystone = client.Client(**kwargs)
    self.assertIsInstance(keystone, v2_client.Client)
    return keystone