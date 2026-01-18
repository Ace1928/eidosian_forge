import uuid
from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.network.v2 import port as _port
from openstack.tests import fakes
from openstack.tests.unit import base
def test_inspect_machine_fail_associated(self):
    self.fake_baremetal_node['provision_state'] = 'available'
    self.fake_baremetal_node['instance_uuid'] = '1234'
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='nodes', append=[self.fake_baremetal_node['uuid']]), json=self.fake_baremetal_node)])
    self.assertRaisesRegex(exceptions.SDKException, 'associated with an instance', self.cloud.inspect_machine, self.fake_baremetal_node['uuid'], wait=True, timeout=1)
    self.assert_calls()