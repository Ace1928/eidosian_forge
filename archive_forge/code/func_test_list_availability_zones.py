import uuid
from openstack.tests.unit import base
def test_list_availability_zones(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('shared-file-system', 'public', append=['v2', 'availability-zones']), json={'availability_zones': [MANILA_AZ_DICT]})])
    az_list = self.cloud.list_share_availability_zones()
    self.assertEqual(len(az_list), 1)
    self.assertEqual(MANILA_AZ_DICT['id'], az_list[0].id)
    self.assertEqual(MANILA_AZ_DICT['name'], az_list[0].name)
    self.assertEqual(MANILA_AZ_DICT['created_at'], az_list[0].created_at)
    self.assertEqual(MANILA_AZ_DICT['updated_at'], az_list[0].updated_at)
    self.assert_calls()