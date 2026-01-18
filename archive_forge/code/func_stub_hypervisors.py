from novaclient.tests.unit.fixture_data import agents as data
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import agents
def stub_hypervisors(self, hypervisor='kvm'):
    get_os_agents = {'agents': [{'hypervisor': hypervisor, 'os': 'win', 'architecture': 'x86', 'version': '7.0', 'url': 'xxx://xxxx/xxx/xxx', 'md5hash': 'add6bb58e139be103324d04d82d8f545', 'id': 1}, {'hypervisor': hypervisor, 'os': 'linux', 'architecture': 'x86', 'version': '16.0', 'url': 'xxx://xxxx/xxx/xxx1', 'md5hash': 'add6bb58e139be103324d04d82d8f546', 'id': 2}]}
    headers = {'Content-Type': 'application/json', 'x-openstack-request-id': fakes.FAKE_REQUEST_ID}
    self.requests_mock.get(self.data_fixture.url(), json=get_os_agents, headers=headers)