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
def test_direct_discovering_with_no_data(self):
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, status_code=400)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    self.assertRaises(exceptions.BadRequest, discover.get_discovery, s, self.TEST_COMPUTE_ADMIN)