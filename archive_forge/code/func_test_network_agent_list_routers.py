import uuid
from openstackclient.tests.functional.network.v2 import common
def test_network_agent_list_routers(self):
    """Add agent to router, list agents on router, delete."""
    if not self.is_extension_enabled('l3_agent_scheduler'):
        self.skipTest('No l3_agent_scheduler extension present')
    name = uuid.uuid4().hex
    cmd_output = self.openstack('router create %s' % name, parse_output=True)
    self.addCleanup(self.openstack, 'router delete %s' % name)
    router_id = cmd_output['id']
    cmd_output = self.openstack('network agent list --agent-type l3', parse_output=True)
    self.assertTrue(cmd_output)
    agent_id = cmd_output[0]['ID']
    self.openstack('network agent add router --l3 %s %s' % (agent_id, router_id))
    cmd_output = self.openstack('network agent list --router %s' % router_id, parse_output=True)
    agent_ids = [x['ID'] for x in cmd_output]
    self.assertIn(agent_id, agent_ids)
    self.openstack('network agent remove router --l3 %s %s' % (agent_id, router_id))
    cmd_output = self.openstack('network agent list --router %s' % router_id, parse_output=True)
    agent_ids = [x['ID'] for x in cmd_output]
    self.assertNotIn(agent_id, agent_ids)