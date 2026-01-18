from novaclient.tests.unit.fixture_data import agents as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import agents
def test_list_agents_with_hypervisor(self):
    self.stub_hypervisors('xen')
    ags = self.cs.agents.list('xen')
    self.assert_called('GET', '/os-agents?hypervisor=xen')
    self.assert_request_id(ags, fakes.FAKE_REQUEST_ID_LIST)
    for a in ags:
        self.assertIsInstance(a, agents.Agent)
        self.assertEqual('xen', a.hypervisor)