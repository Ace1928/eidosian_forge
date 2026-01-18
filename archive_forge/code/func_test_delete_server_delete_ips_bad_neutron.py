import uuid
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_delete_server_delete_ips_bad_neutron(self):
    """
        Test that deleting server with a borked neutron doesn't bork
        """
    server = fakes.make_fake_server('1234', 'porky', 'ACTIVE')
    self.register_uris([self.get_nova_discovery_mock_dict(), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'porky']), status_code=404), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', 'detail'], qs_elements=['name=porky']), json={'servers': [server]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips'], qs_elements=['floating_ip_address=172.24.5.5']), complete_qs=True, status_code=404), dict(method='DELETE', uri=self.get_mock_url('compute', 'public', append=['servers', '1234'])), dict(method='GET', uri=self.get_mock_url('compute', 'public', append=['servers', '1234']), status_code=404)])
    self.assertTrue(self.cloud.delete_server('porky', wait=True, delete_ips=True))
    self.assert_calls()