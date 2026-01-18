from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_hypervisor_uptime(self):
    expected = {'id': self.data_fixture.hyper_id_1, 'hypervisor_hostname': 'hyper1', 'uptime': 'fake uptime', 'state': 'up', 'status': 'enabled'}
    result = self.cs.hypervisors.uptime(self.data_fixture.hyper_id_1)
    self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/os-hypervisors/%s' % self.data_fixture.hyper_id_1)
    self.compare_to_expected(expected, result)