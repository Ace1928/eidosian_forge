import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_machine_by_mac(self):
    mac_address = '00:01:02:03:04:05'
    node_uuid = self.fake_baremetal_node['uuid']
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail'], qs_elements=['address=%s' % mac_address]), json={'ports': [{'address': mac_address, 'node_uuid': node_uuid}]}), dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    machine = self.cloud.get_machine_by_mac(mac_address)
    self.assertEqual(machine['uuid'], self.fake_baremetal_node['uuid'])
    self.assert_calls()