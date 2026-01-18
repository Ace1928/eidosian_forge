from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_delete_subnet_multiple_using_id(self):
    port_name = 'port-name'
    port1 = dict(id='123', name=port_name)
    port2 = dict(id='456', name=port_name)
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', port1['id']]), json={'ports': [port1, port2]}), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', port1['id']]), json={})])
    self.assertTrue(self.cloud.delete_port(name_or_id=port1['id']))
    self.assert_calls()