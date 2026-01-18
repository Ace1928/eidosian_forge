from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_default_domain_not_set(self):
    config = {'identity_api_version': '3', 'auth_type': 'v3oidcpassword', 'default_domain': 'default', 'auth': {'username': 'fred', 'project_id': 'id'}}
    ret_config = self.cloud._auth_default_domain(config)
    self.assertEqual('v3oidcpassword', ret_config['auth_type'])
    self.assertEqual('default', ret_config['default_domain'])
    self.assertEqual('fred', ret_config['auth']['username'])
    self.assertNotIn('project_domain_id', ret_config['auth'])
    self.assertNotIn('user_domain_id', ret_config['auth'])