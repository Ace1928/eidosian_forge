from openstack import exceptions
from openstack.tests import fakes
from openstack.tests.unit import base
def test_get_bad_default_max_microversion(self):
    self.cloud.config.config['compute_default_microversion'] = '2.61'
    self.assertRaises(exceptions.ConfigException, self.cloud.get_server, 'doesNotExist')
    self.assert_calls()