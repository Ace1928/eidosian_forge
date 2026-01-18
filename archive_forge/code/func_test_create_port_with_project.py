from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_create_port_with_project(self):
    self.mock_neutron_port_create_rep['port'].update({'project_id': 'test-project-id'})
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports']), json=self.mock_neutron_port_create_rep, validate=dict(json={'port': {'network_id': 'test-net-id', 'project_id': 'test-project-id', 'name': 'test-port-name', 'admin_state_up': True}}))])
    port = self.cloud.create_port(network_id='test-net-id', name='test-port-name', admin_state_up=True, project_id='test-project-id')
    self._compare_ports(self.mock_neutron_port_create_rep['port'], port)
    self.assert_calls()