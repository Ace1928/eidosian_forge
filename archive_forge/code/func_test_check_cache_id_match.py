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
def test_check_cache_id_match(self):
    a = self.create_auth_plugin()
    b = self.create_auth_plugin()
    self.assertIsNot(a, b)
    self.assertIsNone(a.get_auth_state())
    self.assertIsNone(b.get_auth_state())
    a_id = a.get_cache_id()
    b_id = b.get_cache_id()
    self.assertIsNotNone(a_id)
    self.assertIsNotNone(b_id)
    self.assertEqual(a_id, b_id)