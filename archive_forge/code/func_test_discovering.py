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
def test_discovering(self):
    disc = fixture.DiscoveryList(v2=False, v3=False)
    disc.add_nova_microversion(href=self.TEST_COMPUTE_ADMIN, id='v2.1', status='CURRENT', min_version='2.1', version='2.38')
    self.stub_url('GET', [], base_url=self.TEST_COMPUTE_ADMIN, json=disc)
    body = 'SUCCESS'
    self.stub_url('GET', ['path'], text=body, base_url=self.TEST_COMPUTE_ADMIN)
    a = self.create_auth_plugin()
    s = session.Session(auth=a)
    resp = s.get('/path', endpoint_filter={'service_type': 'compute', 'interface': 'admin', 'version': '2.1'})
    self.assertEqual(200, resp.status_code)
    self.assertEqual(body, resp.text)
    new_body = 'SC SUCCESS'
    self.stub_url('GET', ['path'], base_url=self.TEST_COMPUTE_ADMIN, text=new_body)
    resp = s.get('/path', endpoint_filter={'service_type': 'compute', 'interface': 'admin'})
    self.assertEqual(200, resp.status_code)
    self.assertEqual(new_body, resp.text)