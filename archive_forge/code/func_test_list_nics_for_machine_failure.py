from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_nics_for_machine_failure(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail'], qs_elements=['node_uuid=%s' % self.fake_baremetal_node['uuid']]), status_code=400)])
    self.assertRaises(exceptions.SDKException, self.cloud.list_nics_for_machine, self.fake_baremetal_node['uuid'])
    self.assert_calls()