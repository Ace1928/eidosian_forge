from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_floating_ip_by_id(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips', '1']), json={'floating_ip': self.mock_floating_ip_list_rep[0]})])
    floating_ip = self.cloud.get_floating_ip_by_id(id='1')
    self.assertIsInstance(floating_ip, dict)
    self.assertEqual('203.0.113.1', floating_ip['floating_ip_address'])
    self.assert_calls()