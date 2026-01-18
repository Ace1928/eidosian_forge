from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_conflicting_implied_and_direct(self):
    self.cloud.config.config['compute_default_microversion'] = '2.7'
    self.cloud.config.config['compute_api_version'] = '2.13'
    self.assertRaises(exceptions.ConfigException, self.cloud.get_server)
    self.assertEqual(0, len(self.adapter.request_history))