import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_environment_update(self):
    env = self.environment_create('env.yaml')
    env_name = self.get_field_value(env, 'Name')
    env_desc = self.get_field_value(env, 'Description')
    env_created_at = self.get_field_value(env, 'Created at')
    env_updated_at = self.get_field_value(env, 'Updated at')
    self.assertIsNotNone(env_created_at)
    self.assertEqual('None', env_updated_at)
    self.create_file('env_upd.yaml', 'name: env\ndescription: Updated env\nvariables:\n  var: "value"')
    env = self.mistral_admin('environment-update', params='env_upd.yaml')
    self.assertTableStruct(env, ['Field', 'Value'])
    updated_env_name = self.get_field_value(env, 'Name')
    updated_env_desc = self.get_field_value(env, 'Description')
    updated_env_created_at = self.get_field_value(env, 'Created at')
    updated_env_updated_at = self.get_field_value(env, 'Updated at')
    self.assertEqual(env_name, updated_env_name)
    self.assertNotEqual(env_desc, updated_env_desc)
    self.assertEqual('Updated env', updated_env_desc)
    self.assertEqual(env_created_at.split('.')[0], updated_env_created_at)
    self.assertIsNotNone(updated_env_updated_at)