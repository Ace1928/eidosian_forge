import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_dhcp_agent(self):
    if not self.haz_network:
        self.skipTest('No Network service present')
    if not self.is_extension_enabled('agent'):
        self.skipTest('No dhcp_agent_scheduler extension present')
    if not self.is_extension_enabled('dhcp_agent_scheduler'):
        self.skipTest('No dhcp_agent_scheduler extension present')
    cmd_output = self.openstack('network agent list --agent-type dhcp', parse_output=True)
    if not cmd_output:
        self.skipTest('No agents with type=dhcp available')
    agent_id = cmd_output[0]['ID']
    name1 = uuid.uuid4().hex
    cmd_output = self.openstack('network create --description aaaa %s' % name1, parse_output=True)
    self.addCleanup(self.openstack, 'network delete %s' % name1)
    network_id = cmd_output['id']
    self.openstack('network agent add network --dhcp %s %s' % (agent_id, network_id))
    cmd_output = self.openstack('network list --agent %s' % agent_id, parse_output=True)
    self.openstack('network agent remove network --dhcp %s %s' % (agent_id, network_id))
    col_name = [x['ID'] for x in cmd_output]
    self.assertIn(network_id, col_name)