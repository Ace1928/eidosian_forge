import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_environment_create(self):
    env = self.mistral_admin('environment-create', params='env.yaml')
    env_name = self.get_field_value(env, 'Name')
    env_desc = self.get_field_value(env, 'Description')
    self.assertTableStruct(env, ['Field', 'Value'])
    envs = self.mistral_admin('environment-list')
    self.assertIn(env_name, [en['Name'] for en in envs])
    self.assertIn(env_desc, [en['Description'] for en in envs])
    self.mistral_admin('environment-delete', params=env_name)
    envs = self.mistral_admin('environment-list')
    self.assertNotIn(env_name, [en['Name'] for en in envs])