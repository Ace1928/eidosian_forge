from novaclient.tests.unit.fixture_data import agents as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import agents
def test_agents_create(self):
    ag = self.cs.agents.create('win', 'x86', '7.0', '/xxx/xxx/xxx', 'add6bb58e139be103324d04d82d8f546', 'xen')
    self.assert_request_id(ag, fakes.FAKE_REQUEST_ID_LIST)
    body = {'agent': {'url': '/xxx/xxx/xxx', 'hypervisor': 'xen', 'md5hash': 'add6bb58e139be103324d04d82d8f546', 'version': '7.0', 'architecture': 'x86', 'os': 'win'}}
    self.assert_called('POST', '/os-agents', body)
    self.assertEqual(1, ag._info.copy()['id'])