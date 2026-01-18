from tempest.lib import exceptions
from ironicclient.tests.functional.osc.v1 import base
def test_list_minimal_fields(self):
    headers = ['Instance UUID', 'Name', 'UUID']
    fields = ['instance_uuid', 'name', 'uuid']
    node_list = self.openstack('baremetal node list --fields {}'.format(' '.join(fields)))
    nodes_list_headers = self._get_table_headers(node_list)
    self.assertEqual(headers, nodes_list_headers)