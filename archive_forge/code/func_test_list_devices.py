import copy
import uuid
from openstack.tests.unit import base
def test_list_devices(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('accelerator', 'public', append=['v2', 'devices']), json={'devices': [DEV_DICT]})])
    dev_list = self.cloud.list_devices()
    self.assertEqual(len(dev_list), 1)
    self.assertEqual(dev_list[0].id, DEV_DICT['id'])
    self.assertEqual(dev_list[0].uuid, DEV_DICT['uuid'])
    self.assertEqual(dev_list[0].name, DEV_DICT['name'])
    self.assertEqual(dev_list[0].type, DEV_DICT['type'])
    self.assertEqual(dev_list[0].vendor, DEV_DICT['vendor'])
    self.assertEqual(dev_list[0].model, DEV_DICT['model'])
    self.assertEqual(dev_list[0].std_board_info, DEV_DICT['std_board_info'])
    self.assertEqual(dev_list[0].vendor_board_info, DEV_DICT['vendor_board_info'])
    self.assert_calls()