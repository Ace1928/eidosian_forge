import copy
import uuid
from openstack.tests.unit import base
def test_list_accelerator_requests(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'accelerator_requests']), json={'arqs': [ARQ_DICT]})])
    arq_list = self.cloud.list_accelerator_requests()
    self.assertEqual(len(arq_list), 1)
    self.assertEqual(arq_list[0].uuid, ARQ_DICT['uuid'])
    self.assertEqual(arq_list[0].device_profile_name, ARQ_DICT['device_profile_name'])
    self.assertEqual(arq_list[0].device_profile_group_id, ARQ_DICT['device_profile_group_id'])
    self.assertEqual(arq_list[0].device_rp_uuid, ARQ_DICT['device_rp_uuid'])
    self.assertEqual(arq_list[0].instance_uuid, ARQ_DICT['instance_uuid'])
    self.assertEqual(arq_list[0].attach_handle_type, ARQ_DICT['attach_handle_type'])
    self.assertEqual(arq_list[0].attach_handle_info, ARQ_DICT['attach_handle_info'])
    self.assert_calls()