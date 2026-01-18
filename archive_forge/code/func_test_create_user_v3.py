import uuid
import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_create_user_v3(self):
    user_data = self._get_user_data(domain_id=uuid.uuid4().hex, description=self.getUniqueString('description'))
    self.register_uris([dict(method='POST', uri=self._get_keystone_mock_url(resource='users'), status_code=200, json=user_data.json_response, validate=dict(json=user_data.json_request))])
    user = self.cloud.create_user(name=user_data.name, email=user_data.email, password=user_data.password, description=user_data.description, domain_id=user_data.domain_id)
    self.assertEqual(user_data.name, user.name)
    self.assertEqual(user_data.email, user.email)
    self.assertEqual(user_data.description, user.description)
    self.assertEqual(user_data.user_id, user.id)
    self.assert_calls()