import testtools
from testtools import matchers
from openstack import exceptions
from openstack.tests.unit import base
def test_delete_role_by_id(self):
    role_data = self._get_role_data()
    self.register_uris([dict(method='GET', uri=self.get_mock_url(), status_code=200, json={'roles': [role_data.json_response['role']]}), dict(method='DELETE', uri=self.get_mock_url(append=[role_data.role_id]), status_code=204)])
    role = self.cloud.delete_role(role_data.role_id)
    self.assertThat(role, matchers.Equals(True))
    self.assert_calls()