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
def test_discovery_uses_session_cache(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_nova_microversion(href=self.TEST_COMPUTE_ADMIN, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
    resps = [{'json': disc}, {'status_code': 500}]
    self.requests_mock.get(self.TEST_COMPUTE_ADMIN, resps)
    body = 'SUCCESS'
    self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=body)
    filter = {'service_type': 'compute', 'interface': 'admin', 'version': '2.1'}
    sess = session.Session()
    sess.get('/path', auth=self.create_auth_plugin(), endpoint_filter=filter)
    self.assertIn(self.TEST_COMPUTE_ADMIN, sess._discovery_cache.keys())
    a = self.create_auth_plugin()
    b = self.create_auth_plugin()
    for auth in (a, b):
        resp = sess.get('/path', auth=auth, endpoint_filter=filter)
        self.assertEqual(200, resp.status_code)
        self.assertEqual(body, resp.text)