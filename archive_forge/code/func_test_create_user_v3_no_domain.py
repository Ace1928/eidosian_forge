import uuid
import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_create_user_v3_no_domain(self):
    user_data = self._get_user_data(domain_id=uuid.uuid4().hex, email='test@example.com')
    with testtools.ExpectedException(exceptions.SDKException, 'User or project creation requires an explicit domain_id argument.'):
        self.cloud.create_user(name=user_data.name, email=user_data.email, password=user_data.password)