from novaclient import api_versions
from novaclient import exceptions
from novaclient.tests.unit.fixture_data import client
from novaclient.tests.unit.fixture_data import hypervisors as data
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
def test_hypervisor_get(self):
    expected = dict(id=self.data_fixture.hyper_id_1, service=dict(id=self.data_fixture.service_id_1, host='compute1'), vcpus=4, memory_mb=10 * 1024, local_gb=250, vcpus_used=2, memory_mb_used=5 * 1024, local_gb_used=125, hypervisor_type='xen', hypervisor_version=3, hypervisor_hostname='hyper1', free_ram_mb=5 * 1024, free_disk_gb=125, current_workload=2, running_vms=2, cpu_info='cpu_info', disk_available_least=100, state='up', status='enabled')
    if self.cs.api_version >= api_versions.APIVersion('2.88'):
        del expected['current_workload']
        del expected['disk_available_least']
        del expected['free_ram_mb']
        del expected['free_disk_gb']
        del expected['local_gb']
        del expected['local_gb_used']
        del expected['memory_mb']
        del expected['memory_mb_used']
        del expected['running_vms']
        del expected['vcpus']
        del expected['vcpus_used']
        expected['uptime'] = 'fake uptime'
    result = self.cs.hypervisors.get(self.data_fixture.hyper_id_1)
    self.assert_request_id(result, fakes.FAKE_REQUEST_ID_LIST)
    self.assert_called('GET', '/os-hypervisors/%s' % self.data_fixture.hyper_id_1)
    self.compare_to_expected(expected, result)