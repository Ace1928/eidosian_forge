import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_update_with_id(self):
    acts = self.action_create(self.act_def)
    created_action = self.get_item_info(get_from=acts, get_by='Name', value='greeting')
    action_id = created_action['ID']
    params = '{0} --id {1}'.format(self.act_tag_def, action_id)
    acts = self.mistral_admin('action-update', params=params)
    updated_action = self.get_item_info(get_from=acts, get_by='ID', value=action_id)
    self.assertEqual(created_action['Created at'].split('.')[0], updated_action['Created at'])
    self.assertEqual(created_action['Name'], updated_action['Name'])
    self.assertNotEqual(created_action['Updated at'], updated_action['Updated at'])