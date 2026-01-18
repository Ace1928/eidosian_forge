from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_delete_env_from_another_tenant(self):
    env = self.environment_create(self.env_file)
    env_name = self.get_field_value(env, 'Name')
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'environment-delete', params=env_name)