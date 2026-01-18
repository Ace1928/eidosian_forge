import ddt
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_create_traits(self):
    """Check baremetal allocation create command with traits.

        Test steps:
        1) Create baremetal allocation with specified traits.
        2) Check that allocation successfully created.
        """
    allocation_info = self.allocation_create(params='--trait CUSTOM_1 --trait CUSTOM_2')
    self.assertTrue(allocation_info['resource_class'])
    self.assertEqual(allocation_info['state'], 'allocating')
    self.assertIn('CUSTOM_1', allocation_info['traits'])
    self.assertIn('CUSTOM_2', allocation_info['traits'])