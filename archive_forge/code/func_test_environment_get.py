import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_environment_get(self):
    env = self.environment_create('env.yaml')
    env_name = self.get_field_value(env, 'Name')
    env_desc = self.get_field_value(env, 'Description')
    env = self.mistral_admin('environment-get', params=env_name)
    fetched_env_name = self.get_field_value(env, 'Name')
    fetched_env_desc = self.get_field_value(env, 'Description')
    self.assertTableStruct(env, ['Field', 'Value'])
    self.assertEqual(env_name, fetched_env_name)
    self.assertEqual(env_desc, fetched_env_desc)