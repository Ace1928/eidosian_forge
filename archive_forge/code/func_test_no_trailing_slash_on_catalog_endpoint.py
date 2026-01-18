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
def test_no_trailing_slash_on_catalog_endpoint(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_nova_microversion(href=self.TEST_COMPUTE_PUBLIC, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    self.requests_mock.get(self.TEST_COMPUTE_PUBLIC, json=disc)
    s.get_endpoint_data(service_type='compute', interface='public', min_version='2.1', max_version='2.latest')
    self.assertFalse(self.requests_mock.request_history[-1].url.endswith('/'))