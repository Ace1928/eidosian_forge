from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_select_default_plugin_password(self):
    config = {'username': 'fred', 'user_id': 'fr3d'}
    ret_config = self.cloud._auth_select_default_plugin(config)
    self.assertEqual('password', ret_config['auth_type'])
    self.assertEqual('fred', ret_config['username'])
    self.assertEqual('fr3d', ret_config['user_id'])