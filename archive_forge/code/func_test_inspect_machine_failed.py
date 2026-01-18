import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_inspect_machine_failed(self):
    inspecting_node = self.fake_baremetal_node.copy()
    self.fake_baremetal_node['provision_state'] = 'inspect failed'
    self.fake_baremetal_node['last_error'] = 'kaboom!'
    inspecting_node['provision_state'] = 'inspecting'
    finished_node = self.fake_baremetal_node.copy()
    finished_node['provision_state'] = 'manageable'
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node), dict(method='PUT', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid'], 'states', 'provision']), validate=dict(json={'target': 'inspect'})), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=inspecting_node), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=finished_node)])
    self.cloud.inspect_machine(self.fake_baremetal_node['uuid'])
    self.assert_calls()