from openstack.tests import fakes
from openstack.tests.unit import base
def test_attach_ip_to_server(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('compute', append=['os-floating-ips']), json={'floating_ips': self.mock_floating_ip_list_rep}), dict(method='POST', uri=self.get_mock_url('compute', append=['servers', self.fake_server['id'], 'action']), validate=dict(json={'addFloatingIp': {'address': '203.0.113.1', 'fixed_address': '192.0.2.129'}}))])
    self.cloud._attach_ip_to_server(server=self.fake_server, floating_ip=self.cloud._normalize_floating_ip(self.mock_floating_ip_list_rep[0]), fixed_address='192.0.2.129')
    self.assert_calls()