import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_server_delete_ips(self):
    """
        Test that deleting server and fips works
        """
    server = fakes.make_fake_server('1234', 'porky', 'ACTIVE')
    fip_id = uuid.uuid4().hex
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'porky']), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=porky']), json={'servers': [server]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips'], qs_elements=['floating_ip_address=172.24.5.5']), complete_qs=True, json={'floatingips': [{'router_id': 'd23abc8d-2991-4a55-ba98-2aaea84cc72f', 'tenant_id': '4969c491a3c74ee4af974e6d800c62de', 'floating_network_id': '376da547-b977-4cfe-9cba7', 'fixed_ip_address': '10.0.0.4', 'floating_ip_address': '172.24.5.5', 'port_id': 'ce705c24-c1ef-408a-bda3-7bbd946164ac', 'id': fip_id, 'status': 'ACTIVE'}]}), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips', fip_id])), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips']), complete_qs=True, json={'floatingips': []}), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', '1234'])), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), status_code=404)])
    self.assertTrue(self.cloud.delete_server('porky', wait=True, delete_ips=True))
    self.assert_calls()