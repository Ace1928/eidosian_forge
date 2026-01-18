from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_config_hook_default(self):
    config = {}
    ret_config = self.cloud.auth_config_hook(config)
    self.assertEqual('password', ret_config['auth_type'])