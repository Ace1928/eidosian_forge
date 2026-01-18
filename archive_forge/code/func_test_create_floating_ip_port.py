import copy
import datetime
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
from openstack import utils
def test_create_floating_ip_port(self):
    self.register_uris([dict(method='GET', uri='https://network.example.com/v2.0/networks/my-network', status_code=404), dict(method='GET', uri='https://network.example.com/v2.0/networks?name=my-network', json={'networks': [self.mock_get_network_rep]}), dict(method='POST', uri='https://network.example.com/v2.0/floatingips', json=self.mock_floating_ip_port_rep, validate=dict(json={'floatingip': {'floating_network_id': 'my-network-id', 'port_id': 'ce705c24-c1ef-408a-bda3-7bbd946164ac'}}))])
    ip = self.cloud.create_floating_ip(network='my-network', port='ce705c24-c1ef-408a-bda3-7bbd946164ac')
    self.assertEqual(self.mock_floating_ip_new_rep['floatingip']['floating_ip_address'], ip['floating_ip_address'])
    self.assert_calls()