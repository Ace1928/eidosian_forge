import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_delete_network(self):
    network_id = 'test-net-id'
    network_name = 'network'
    network = {'id': network_id, 'name': network_name}
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', network_name]), status_code=404), dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks'], qs_elements=['name=%s' % network_name]), json={'networks': [network]}), dict(method='DELETE', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks', network_id]), json={})])
    self.assertTrue(self.cloud.delete_network(network_name))
    self.assert_calls()