import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_node_set_provision_state_with_retries(self):
    deploy_node = self.fake_baremetal_node.copy()
    deploy_node['provision_state'] = 'deploying'
    active_node = self.fake_baremetal_node.copy()
    active_node['provision_state'] = 'active'
    self.register_uris([dict(method='PUT', status_code=409, uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'active', 'configdrive': 'http://host/file'})), dict(method='PUT', status_code=503, uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'active', 'configdrive': 'http://host/file'})), dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'active', 'configdrive': 'http://host/file'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    self.cloud.node_set_provision_state(self.fake_baremetal_node['uuid'], 'active', configdrive='http://host/file')
    self.assert_calls()