from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_delete_action_from_another_tenant(self):
    act = self.action_create(self.act_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'action-delete', params=act[0]['Name'])