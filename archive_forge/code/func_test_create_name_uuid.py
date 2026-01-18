import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_create_name_uuid(self):
    """Check baremetal allocation create command with name and UUID.

        Test steps:
        1) Create baremetal allocation with specified name and UUID.
        2) Check that allocation successfully created.
        """
    uuid = data_utils.rand_uuid()
    name = data_utils.rand_name('baremetal-allocation')
    allocation_info = self.allocation_create(params='--uuid {0} --name {1}'.format(uuid, name))
    self.assertEqual(allocation_info['uuid'], uuid)
    self.assertEqual(allocation_info['name'], name)
    self.assertTrue(allocation_info['resource_class'])
    self.assertEqual(allocation_info['state'], 'allocating')
    allocation_list = self.allocation_list()
    self.assertIn(uuid, [x['UUID'] for x in allocation_list])
    self.assertIn(name, [x['Name'] for x in allocation_list])