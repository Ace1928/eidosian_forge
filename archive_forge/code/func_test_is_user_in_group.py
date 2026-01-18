import uuid
import testtools
from openstack import exceptions
from openstack.tests.unit import base
def test_is_user_in_group(self):
    user_data = self._get_user_data()
    group_data = self._get_group_data()
    self.register_uris([dict(method='GET', uri=self._get_keystone_mock_url(resource='users'), status_code=200, json=self._get_user_list(user_data)), dict(method='GET', uri=self._get_keystone_mock_url(resource='groups'), status_code=200, json={'groups': [group_data.json_response['group']]}), dict(method='HEAD', uri=self._get_keystone_mock_url(resource='groups', append=[group_data.group_id, 'users', user_data.user_id]), status_code=204)])
    self.assertTrue(self.cloud.is_user_in_group(user_data.user_id, group_data.group_id))
    self.assert_calls()