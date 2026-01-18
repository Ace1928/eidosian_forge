import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_event_trigger_create_delete(self):
    trigger = self.mistral_admin('event-trigger-create', params='trigger %s dummy_exchange dummy_topic event.dummy {}' % self.wf_id)
    self.assertTableStruct(trigger, ['Field', 'Value'])
    tr_id = self.get_field_value(trigger, 'ID')
    tr_name = self.get_field_value(trigger, 'Name')
    wf_id = self.get_field_value(trigger, 'Workflow ID')
    created_at = self.get_field_value(trigger, 'Created at')
    self.assertEqual('trigger', tr_name)
    self.assertEqual(self.wf_id, wf_id)
    self.assertIsNotNone(created_at)
    triggers = self.mistral_admin('event-trigger-list')
    self.assertIn(tr_name, [tr['Name'] for tr in triggers])
    self.assertIn(wf_id, [tr['Workflow ID'] for tr in triggers])
    self.mistral('event-trigger-delete', params=tr_id)
    triggers = self.mistral_admin('event-trigger-list')
    self.assertNotIn(tr_name, [tr['Name'] for tr in triggers])