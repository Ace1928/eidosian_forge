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
def test_latest_version_gets_latest_version(self):
    common_disc = fixture.DiscoveryList(href=self.BASE_URL)
    v2_m = self.stub_url('GET', base_url=self.BASE_URL, status_code=200, json=common_disc)
    token = fixture.V2Token()
    service = token.add_service(self.IDENTITY)
    service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
    self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
    v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
    sess = session.Session(auth=v2_auth)
    self.assertFalse(v2_m.called)
    endpoint = sess.get_endpoint(service_type=self.IDENTITY, version='latest')
    self.assertTrue(v2_m.called)
    self.assertEqual(self.V3_URL, endpoint)
    endpoint = sess.get_endpoint(service_type=self.IDENTITY, max_version='latest')
    self.assertEqual(self.V3_URL, endpoint)
    endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='latest')
    self.assertEqual(self.V3_URL, endpoint)
    endpoint = sess.get_endpoint(service_type=self.IDENTITY, min_version='latest', max_version='latest')
    self.assertEqual(self.V3_URL, endpoint)
    self.assertRaises(ValueError, sess.get_endpoint, service_type=self.IDENTITY, min_version='latest', max_version='3.0')