from osc_lib.cli import client_config
from osc_lib.tests import utils
def test_auth_v2_ignore_v3(self):
    config = {'cloud': 'testcloud', 'identity_api_version': '2', 'auth_type': 'v2password', 'auth': {'username': 'fred', 'project_id': 'id', 'project_domain_id': 'bad'}}
    ret_config = self.cloud._auth_v2_ignore_v3(config)
    self.assertEqual('fred', ret_config['auth']['username'])
    self.assertNotIn('project_domain_id', ret_config['auth'])