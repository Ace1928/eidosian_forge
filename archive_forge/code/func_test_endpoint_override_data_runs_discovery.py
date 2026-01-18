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
def test_endpoint_override_data_runs_discovery(self):
    common_disc = fixture.DiscoveryList(v2=False, v3=False)
    common_disc.add_microversion(href=self.OTHER_URL, id='v2.1', min_version='2.1', max_version='2.35')
    common_m = self.stub_url('GET', base_url=self.OTHER_URL, status_code=200, json=common_disc)
    token = fixture.V2Token()
    service = token.add_service(self.IDENTITY)
    service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
    self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
    v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
    sess = session.Session(auth=v2_auth)
    data = sess.get_endpoint_data(endpoint_override=self.OTHER_URL, service_type=self.IDENTITY, interface='public', min_version=(2, 0), max_version=(2, discover.LATEST))
    self.assertTrue(common_m.called)
    self.assertEqual(self.OTHER_URL, data.service_url)
    self.assertEqual(self.OTHER_URL, data.catalog_url)
    self.assertEqual(self.OTHER_URL, data.url)
    self.assertEqual((2, 1), data.min_microversion)
    self.assertEqual((2, 35), data.max_microversion)
    self.assertEqual((2, 1), data.api_version)