import json
import ddt
from tempest.lib.common.utils import data_utils
from ironicclient.tests.functional.osc.v1 import base
def test_baremetal_node_maintenance_set_unset_reason(self):
    """Check baremetal node maintenance set command.

        Test steps:
        1) Create baremetal node in setUp.
        2) Check initial maintenance reason is None.
        3) Set maintenance status for node with reason.
        4) Check maintenance reason of node equals to expected value.
           Also check maintenance status.
        5) Unset maintenance status for node. Recheck maintenance status.
        6) Check maintenance reason is None. Recheck maintenance status.
        """
    reason = 'Hardware maintenance.'
    show_prop = self.node_show(self.node['name'], ['maintenance_reason', 'maintenance'])
    self.assertIsNone(show_prop['maintenance_reason'])
    self.assertFalse(show_prop['maintenance'])
    self.openstack("baremetal node maintenance set --reason '{0}' {1}".format(reason, self.node['name']))
    show_prop = self.node_show(self.node['name'], ['maintenance_reason', 'maintenance'])
    self.assertEqual(reason, show_prop['maintenance_reason'])
    self.assertTrue(show_prop['maintenance'])
    self.openstack('baremetal node maintenance unset {0}'.format(self.node['name']))
    show_prop = self.node_show(self.node['name'], ['maintenance_reason', 'maintenance'])
    self.assertIsNone(show_prop['maintenance_reason'])
    self.assertFalse(show_prop['maintenance'])