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
def test_endpoint_no_version(self):
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    data = a.get_endpoint(session=s, service_type='compute', interface='admin')
    self.assertEqual(self.TEST_COMPUTE_ADMIN, data)