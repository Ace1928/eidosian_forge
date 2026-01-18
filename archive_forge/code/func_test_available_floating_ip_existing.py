from openstack.tests import fakes
from openstack.tests.unit import base
def test_available_floating_ip_existing(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep[:1]})])
    ip = self.cloud.available_floating_ip(network='nova')
    self.assertEqual(self.mock_floating_ip_list_rep[0]['ip'], ip['floating_ip_address'])
    self.assert_calls()