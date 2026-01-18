import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_cron_trigger_get(self):
    trigger = self.cron_trigger_create('trigger', self.wf_name, '{}', '5 * * * *')
    self.assertTableStruct(trigger, ['Field', 'Value'])
    fetched_tr = self.mistral_admin('cron-trigger-get', params='trigger')
    self.assertTableStruct(trigger, ['Field', 'Value'])
    tr_name = self.get_field_value(fetched_tr, 'Name')
    wf_name = self.get_field_value(fetched_tr, 'Workflow')
    created_at = self.get_field_value(fetched_tr, 'Created at')
    self.assertEqual('trigger', tr_name)
    self.assertEqual(self.wf_name, wf_name)
    self.assertIsNotNone(created_at)