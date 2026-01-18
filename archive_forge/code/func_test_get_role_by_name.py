import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_get_role_by_name(self):
    role_data = self._get_role_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'roles': [role_data.json_response['role']]})])
    role = self.cloud.get_role(role_data.role_name)
    self.assertIsNotNone(role)
    self.assertThat(role.id, matchers.Equals(role_data.role_id))
    self.assertThat(role.name, matchers.Equals(role_data.role_name))
    self.assert_calls()