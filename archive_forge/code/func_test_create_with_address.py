import ddt
from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
def test_create_with_address(self):
    """Check baremetal port group create command with address argument.

        Test steps:
        1) Create baremetal port group in setUp.
        2) Create baremetal port group with specific address argument.
        3) Check address of created port group.
        """
    mac_address = data_utils.rand_mac_address()
    port_group = self.port_group_create(self.node['uuid'], params='{0} --address {1}'.format(self.api_version, mac_address))
    self.assertEqual(mac_address, port_group['address'])