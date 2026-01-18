from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_select_default_plugin_password_v2_int(self):
    config = {'identity_api_version': 2, 'username': 'fred'}
    ret_config = self.cloud._auth_select_default_plugin(config)
    self.assertEqual('v2password', ret_config['auth_type'])
    self.assertEqual('fred', ret_config['username'])