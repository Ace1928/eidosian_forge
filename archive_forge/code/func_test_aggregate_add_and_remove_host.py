import uuid
from openstackclient.tests.functional import base
def test_aggregate_add_and_remove_host(self):
    """Test aggregate add and remove host"""
    cmd_output = self.openstack('host list', parse_output=True)
    host_name = cmd_output[0]['Host Name']
    if '@' in host_name:
        self.skipTest('Skip aggregates in a Nova cells v1 configuration')
    name = uuid.uuid4().hex
    self.addCleanup(self.openstack, 'aggregate delete ' + name)
    self.openstack('aggregate create ' + name)
    cmd_output = self.openstack('aggregate add host ' + name + ' ' + host_name, parse_output=True)
    self.assertIn(host_name, cmd_output['hosts'])
    cmd_output = self.openstack('aggregate remove host ' + name + ' ' + host_name, parse_output=True)
    self.assertNotIn(host_name, cmd_output['hosts'])