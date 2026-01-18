import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_with_mtu(self):
    mtu_size = 1500
    mock_new_network_rep = copy.copy(self.mock_new_network_rep)
    mock_new_network_rep['mtu'] = mtu_size
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'network': mock_new_network_rep}, validate=dict(json={'network': {'admin_state_up': True, 'name': 'netname', 'mtu': mtu_size}}))])
    network = self.cloud.create_network('netname', mtu_size=mtu_size)
    self._compare_networks(mock_new_network_rep, network)
    self.assert_calls()