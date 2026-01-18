import fixtures
from openstack import config
from openstack.config import cloud_region
from openstack import exceptions
from openstack.tests.unit.config import base
def test_have_envvars(self):
    self.useFixture(fixtures.EnvironmentVariable('NOVA_USERNAME', 'nova'))
    self.useFixture(fixtures.EnvironmentVariable('OS_AUTH_URL', 'http://example.com'))
    self.useFixture(fixtures.EnvironmentVariable('OS_USERNAME', 'user'))
    self.useFixture(fixtures.EnvironmentVariable('OS_PASSWORD', 'password'))
    self.useFixture(fixtures.EnvironmentVariable('OS_PROJECT_NAME', 'project'))
    c = config.OpenStackConfig(config_files=[self.cloud_yaml], vendor_files=[self.vendor_yaml])
    cc = c.get_one('envvars')
    self.assertEqual(cc.config['auth']['username'], 'user')