from testtools import matchers
from openstack.tests.unit import base
def test_create_user_without_password(self):
    domain_data = self._get_domain_data()
    user_data = self._get_user_data('myusername', domain_id=domain_data.domain_id)
    user_data._replace(password=None, json_request=user_data.json_request['user'].pop('password'))
    self.register_uris([dict(method='POST', uri=self.get_mock_url(), status_code=200, json=user_data.json_response, validate=dict(json=user_data.json_request))])
    user = self.cloud.create_user(user_data.name, domain_id=domain_data.domain_id)
    self.assertIsNotNone(user)
    self.assertThat(user.name, matchers.Equals(user_data.name))
    self.assert_calls()