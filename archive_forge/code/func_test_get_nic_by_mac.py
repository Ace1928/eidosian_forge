from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_nic_by_mac(self):
    mac = self.fake_baremetal_port['address']
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail'], qs_elements=['address=%s' % mac]), json={'ports': [self.fake_baremetal_port]})])
    return_value = self.cloud.get_nic_by_mac(mac)
    self.assertSubdict(self.fake_baremetal_port, return_value)
    self.assert_calls()