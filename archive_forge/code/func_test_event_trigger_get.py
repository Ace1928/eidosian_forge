import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_event_trigger_get(self):
    trigger = self.event_trigger_create('trigger', self.wf_id, 'dummy_exchange', 'dummy_topic', 'event.dummy.other', '{}')
    self.assertTableStruct(trigger, ['Field', 'Value'])
    ev_tr_id = self.get_field_value(trigger, 'ID')
    fetched_tr = self.mistral_admin('event-trigger-get', params=ev_tr_id)
    self.assertTableStruct(trigger, ['Field', 'Value'])
    tr_name = self.get_field_value(fetched_tr, 'Name')
    wf_id = self.get_field_value(fetched_tr, 'Workflow ID')
    created_at = self.get_field_value(fetched_tr, 'Created at')
    self.assertEqual('trigger', tr_name)
    self.assertEqual(self.wf_id, wf_id)
    self.assertIsNotNone(created_at)