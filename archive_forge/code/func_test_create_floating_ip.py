from openstack.tests import fakes
from openstack.tests.unit import base
def test_create_floating_ip(self):
    self.register_uris([dict(method='POST', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ip': self.mock_floating_ip_list_rep[1]}, validate=dict(json={'pool': 'nova'})), dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips', '2']), json={'floating_ip': self.mock_floating_ip_list_rep[1]})])
    self.cloud.create_floating_ip(network='nova')
    self.assert_calls()