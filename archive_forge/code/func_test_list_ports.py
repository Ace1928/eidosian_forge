from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_list_ports(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports']), json=self.mock_neutron_port_list_rep)])
    ports = self.cloud.list_ports()
    for a, b in zip(self.mock_neutron_port_list_rep['ports'], ports):
        self._compare_ports(a, b)
    self.assert_calls()