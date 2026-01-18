from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_v2_arguments(self):
    config = {'identity_api_version': '2', 'auth_type': 'v2password', 'auth': {'username': 'fred'}}
    ret_config = self.cloud._auth_v2_arguments(config)
    self.assertEqual('fred', ret_config['auth']['username'])
    self.assertNotIn('tenant_id', ret_config['auth'])
    self.assertNotIn('tenant_name', ret_config['auth'])
    config = {'identity_api_version': '3', 'auth_type': 'v3password', 'auth': {'username': 'fred', 'project_id': 'id'}}
    ret_config = self.cloud._auth_v2_arguments(config)
    self.assertEqual('fred', ret_config['auth']['username'])
    self.assertNotIn('tenant_id', ret_config['auth'])
    self.assertNotIn('tenant_name', ret_config['auth'])
    config = {'identity_api_version': '2', 'auth_type': 'v2password', 'auth': {'username': 'fred', 'project_id': 'id'}}
    ret_config = self.cloud._auth_v2_arguments(config)
    self.assertEqual('id', ret_config['auth']['tenant_id'])
    self.assertNotIn('tenant_name', ret_config['auth'])
    config = {'identity_api_version': '2', 'auth_type': 'v2password', 'auth': {'username': 'fred', 'project_name': 'name'}}
    ret_config = self.cloud._auth_v2_arguments(config)
    self.assertNotIn('tenant_id', ret_config['auth'])
    self.assertEqual('name', ret_config['auth']['tenant_name'])