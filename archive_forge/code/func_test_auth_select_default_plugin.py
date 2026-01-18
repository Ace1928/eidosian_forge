from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_select_default_plugin(self):
    config = {'auth_type': 'admin_token'}
    ret_config = self.cloud._auth_select_default_plugin(config)
    self.assertEqual('admin_token', ret_config['auth_type'])