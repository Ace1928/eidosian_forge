import copy
import datetime
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack import utils
def test_create_floating_ip_no_port(self):
    server_port = {'id': 'port-id', 'device_id': 'some-server', 'created_at': datetime.datetime.now().isoformat(), 'fixed_ips': [{'subnet_id': 'subnet-id', 'ip_address': '172.24.4.2'}]}
    floating_ip = {'id': 'floating-ip-id', 'port_id': None}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'networks': [self.mock_get_network_rep]}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'subnets']), json={'subnets': []}), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'ports'], qs_elements=['device_id=some-server']), json={'ports': [server_port]}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'floatingips']), json={'floatingip': floating_ip})])
    self.assertRaises(exceptions.SDKException, self.cloud._neutron_create_floating_ip, server=dict(id='some-server'))
    self.assert_calls()