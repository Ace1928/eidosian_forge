import json
import ddt
from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
def test_create_old_api_version(self):
    """Check baremetal node create command with name and UUID.

        Test steps:
        1) Create baremetal node in setUp.
        2) Create one more baremetal node explicitly with old API version
        3) Check that node successfully created.
        """
    node_info = self.node_create(params='--os-baremetal-api-version 1.5')
    self.assertEqual(node_info['driver'], self.driver_name)
    self.assertEqual(node_info['maintenance'], False)
    self.assertEqual(node_info['provision_state'], 'available')