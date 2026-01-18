import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
def test_get_set_auth_state(self):
    a = self.create_auth_plugin()
    b = self.create_auth_plugin()
    self.assertEqual(a.get_cache_id(), b.get_cache_id())
    s = session.Session()
    a_token = a.get_token(s)
    self.assertEqual(1, self.requests_mock.call_count)
    auth_state = a.get_auth_state()
    self.assertIsNotNone(auth_state)
    b.set_auth_state(auth_state)
    b_token = b.get_token(s)
    self.assertEqual(1, self.requests_mock.call_count)
    self.assertEqual(a_token, b_token)
    self.assertAccessInfoEqual(a.auth_ref, b.auth_ref)