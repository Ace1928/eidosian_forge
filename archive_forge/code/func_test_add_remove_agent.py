from openstack.network.v2 import network
from openstack.tests.functional import base
def test_add_remove_agent(self):
    net = self.AGENT.add_agent_to_network(self.user_cloud.network, network_id=self.NETWORK_ID)
    self._verify_add(net)
    net = self.AGENT.remove_agent_from_network(self.user_cloud.network, network_id=self.NETWORK_ID)
    self._verify_remove(net)