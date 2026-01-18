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
def test_no_reauthenticate(self):
    a = self._create_expired_auth_plugin(reauthenticate=False)
    expired_auth_ref = a.auth_ref
    s = session.Session(auth=a)
    self.assertIs(expired_auth_ref, a.get_access(s))