import copy
from unittest import mock
import testtools
from openstack import exceptions
from openstack.network.v2 import network as _network
from openstack.tests.unit import base
def test_create_network_specific_tenant(self):
    project_id = 'project_id_value'
    mock_new_network_rep = copy.copy(self.mock_new_network_rep)
    mock_new_network_rep['project_id'] = project_id
    self.register_uris([dict(method='POST', uri=self.get_mock_url('network', 'public', append=['v2.0', 'networks']), json={'network': mock_new_network_rep}, validate=dict(json={'network': {'admin_state_up': True, 'name': 'netname', 'project_id': project_id}}))])
    network = self.cloud.create_network('netname', project_id=project_id)
    self._compare_networks(mock_new_network_rep, network)
    self.assert_calls()