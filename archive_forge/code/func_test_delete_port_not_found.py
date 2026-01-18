from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests.unit import base
def test_delete_port_not_found(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports', 'non-existent']), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports'], qs_elements=['name=non-existent']), json={'ports': []})])
    self.assertFalse(self.cloud.delete_port(name_or_id='non-existent'))
    self.assert_calls()