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
def test_endpoint_override_skips_discovery(self):
    token = fixture.V2Token()
    service = token.add_service(self.IDENTITY)
    service.add_endpoint(public=self.V2_URL, admin=self.V2_URL, internal=self.V2_URL)
    self.stub_url('POST', ['tokens'], base_url=self.V2_URL, json=token)
    v2_auth = identity.V2Password(self.V2_URL, username=uuid.uuid4().hex, password=uuid.uuid4().hex)
    sess = session.Session(auth=v2_auth)
    endpoint = sess.get_endpoint(endpoint_override=self.OTHER_URL, service_type=self.IDENTITY, interface='public', version=(3, 0))
    self.assertEqual(self.OTHER_URL, endpoint)