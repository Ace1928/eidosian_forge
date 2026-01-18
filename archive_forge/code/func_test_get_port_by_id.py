from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_get_port_by_id(self):
    fake_port = dict(id='123', name='456')
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', fake_port['id']]), json={'port': fake_port})])
    r = self.cloud.get_port_by_id(fake_port['id'])
    self.assertIsNotNone(r)
    self._compare_ports(fake_port, r)
    self.assert_calls()