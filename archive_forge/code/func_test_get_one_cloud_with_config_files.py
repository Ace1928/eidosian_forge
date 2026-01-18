from os_client_config import cloud_config
from os_client_config import config
from os_client_config import exceptions
from os_client_config.tests import base
import fixtures
def test_get_one_cloud_with_config_files(self):
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml], secure_files=[self.secure_yaml])
    self.assertIsInstance(c.cloud_config, dict)
    self.assertIn('cache', c.cloud_config)
    self.assertIsInstance(c.cloud_config['cache'], dict)
    self.assertIn('max_age', c.cloud_config['cache'])
    self.assertIn('path', c.cloud_config['cache'])
    cc = c.get_one_cloud('_test-cloud_')
    self._assert_cloud_details(cc)
    cc = c.get_one_cloud('_test_cloud_no_vendor')
    self._assert_cloud_details(cc)