import fixtures
from openstack import config
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests.unit.config import base
def test_config_file_override(self):
    self.useFixture(fixtures.EnvironmentVariable('OS_CLIENT_CONFIG_FILE', self.cloud_yaml))
    c = config.OpenStackConfig(config_files=[], vendor_files=[self.vendor_yaml])
    cc = c.get_one('_test-cloud_')
    self._assert_cloud_details(cc)