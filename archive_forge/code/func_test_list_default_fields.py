from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_list_default_fields(self):
    """Test presence of default list table headers."""
    headers = ['UUID', 'Name', 'Instance UUID', 'Power State', 'Provisioning State', 'Maintenance']
    nodes_list = self.openstack('baremetal node list')
    nodes_list_headers = self._get_table_headers(nodes_list)
    self.assertEqual(set(headers), set(nodes_list_headers))