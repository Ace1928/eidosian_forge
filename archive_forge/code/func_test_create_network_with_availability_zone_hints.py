import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_with_availability_zone_hints(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url('network', 'public', append=['v2.0', 'extensions']), json={'extensions': self.enabled_neutron_extensions}), dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'network': self.mock_new_network_rep}, validate=dict(json={'network': {'admin_state_up': True, 'name': 'netname', 'availability_zone_hints': ['nova']}}))])
    network = self.cloud.create_network('netname', availability_zone_hints=['nova'])
    self._compare_networks(self.mock_new_network_rep, network)
    self.assert_calls()