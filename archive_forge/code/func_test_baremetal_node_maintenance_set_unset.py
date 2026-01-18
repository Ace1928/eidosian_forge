import json
import ddt
from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
def test_baremetal_node_maintenance_set_unset(self):
    """Check baremetal node maintenance set command.

        Test steps:
        1) Create baremetal node in setUp.
        2) Check maintenance status of fresh node is False.
        3) Set maintenance status for node.
        4) Check maintenance status of node is True.
        5) Unset maintenance status for node.
        6) Check maintenance status of node is False back.
        """
    show_prop = self.node_show(self.node['name'], ['maintenance'])
    self.assertFalse(show_prop['maintenance'])
    self.openstack('baremetal node maintenance set {0}'.format(self.node['name']))
    show_prop = self.node_show(self.node['name'], ['maintenance'])
    self.assertTrue(show_prop['maintenance'])
    self.openstack('baremetal node maintenance unset {0}'.format(self.node['name']))
    show_prop = self.node_show(self.node['name'], ['maintenance'])
    self.assertFalse(show_prop['maintenance'])