from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_actions_name_uniqueness(self):
    self.action_create(self.act_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-create', params='{0}'.format(self.act_def))
    self.action_create(self.act_def, admin=False)
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'action-create', params='{0}'.format(self.act_def))