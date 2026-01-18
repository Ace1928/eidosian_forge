from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_environment_isolation(self):
    env = self.environment_create(self.env_file)
    env_name = self.get_field_value(env, 'Name')
    envs = self.mistral_admin('environment-list')
    self.assertIn(env_name, [en['Name'] for en in envs])
    alt_envs = self.mistral_alt_user('environment-list')
    self.assertNotIn(env_name, [en['Name'] for en in alt_envs])