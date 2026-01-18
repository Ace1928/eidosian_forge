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
def test_endpoint_data_relative_version(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_v2('v2.0')
    disc.add_v3('v3')
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    data_v2 = a.get_endpoint_data(session=s, service_type='compute', interface='admin', min_version=(2, 0), max_version=(2, discover.LATEST))
    data_v3 = a.get_endpoint_data(session=s, service_type='compute', interface='admin', min_version=(3, 0), max_version=(3, discover.LATEST))
    self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v2.0', data_v2.url)
    self.assertEqual(self.TEST_COMPUTE_ADMIN + '/v3', data_v3.url)