from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_default_domain_use_default(self):
    config = {'identity_api_version': '3', 'auth_type': 'v3password', 'default_domain': 'default', 'auth': {'username': 'fred', 'project_id': 'id'}}
    ret_config = self.cloud._auth_default_domain(config)
    self.assertEqual('v3password', ret_config['auth_type'])
    self.assertEqual('default', ret_config['default_domain'])
    self.assertEqual('fred', ret_config['auth']['username'])
    self.assertEqual('default', ret_config['auth']['project_domain_id'])
    self.assertEqual('default', ret_config['auth']['user_domain_id'])