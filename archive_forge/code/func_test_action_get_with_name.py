import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_get_with_name(self):
    created = self.action_create(self.act_def)
    action_name = created[0]['Name']
    fetched = self.mistral_admin('action-get', params=action_name)
    fetched_action_name = self.get_field_value(fetched, 'Name')
    self.assertEqual(action_name, fetched_action_name)