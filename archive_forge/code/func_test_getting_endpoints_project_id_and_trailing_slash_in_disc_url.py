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
def test_getting_endpoints_project_id_and_trailing_slash_in_disc_url(self):
    disc = fixture.DiscoveryList(href=self.BASE_URL)
    self.stub_url('GET', ['/'], base_url=self.BASE_URL, json=disc)
    token = fixture.V3Token(project_id=self.PROJECT_ID)
    service = token.add_service(self.IDENTITY)
    service.add_endpoint('public', self.V2_URL + '/')
    service.add_endpoint('admin', self.V2_URL + '/')
    kwargs = {'headers': {'X-Subject-Token': self.TEST_TOKEN}}
    self.stub_url('POST', ['auth', 'tokens'], base_url=self.V3_URL, json=token, **kwargs)
    v3_auth = identity.V3Password(self.V3_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
    sess = session.Session(auth=v3_auth)
    endpoint = sess.get_endpoint(service_type=self.IDENTITY, interface='public', version=(3, 0))
    self.assertEqual(self.V3_URL, endpoint)