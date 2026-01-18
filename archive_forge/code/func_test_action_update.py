import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_update(self):
    actions = self.action_create(self.act_def)
    created_action = self.get_item_info(get_from=actions, get_by='Name', value='greeting')
    actions = self.mistral_admin('action-update', params=self.act_def)
    updated_action = self.get_item_info(get_from=actions, get_by='Name', value='greeting')
    self.assertEqual(created_action['Created at'].split('.')[0], updated_action['Created at'])
    self.assertEqual(created_action['Name'], updated_action['Name'])
    self.assertEqual(created_action['Updated at'], updated_action['Updated at'])
    actions = self.mistral_admin('action-update', params=self.act_tag_def)
    updated_action = self.get_item_info(get_from=actions, get_by='Name', value='greeting')
    self.assertEqual('tag, tag1', updated_action['Tags'])
    self.assertEqual(created_action['Created at'].split('.')[0], updated_action['Created at'])
    self.assertEqual(created_action['Name'], updated_action['Name'])
    self.assertNotEqual(created_action['Updated at'], updated_action['Updated at'])