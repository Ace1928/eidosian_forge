from testscenarios import load_tests_apply_scenarios as load_tests  # noqa
from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_list_nics(self):
    self.register_uris([dict(method='GET', uri=self.get_mock_url(resource='ports', append=['detail']), json={'ports': [self.fake_baremetal_port, self.fake_baremetal_port2]})])
    return_value = self.cloud.list_nics()
    self.assertEqual(2, len(return_value))
    self.assertSubdict(self.fake_baremetal_port, return_value[0])
    self.assert_calls()