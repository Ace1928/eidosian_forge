from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_select_default_plugin_password_v3_int(self):
    config = {'identity_api_version': 3, 'username': 'fred'}
    ret_config = self.cloud._auth_select_default_plugin(config)
    self.assertEqual('v3password', ret_config['auth_type'])
    self.assertEqual('fred', ret_config['username'])