from novaclient.tests.unit.fixture_data import agents as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import agents
def test_agents_delete(self):
    ret = self.cs.agents.delete('1')
    self.assert_request_id(ret, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('DELETE', '/os-agents/1')