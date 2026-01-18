from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_cron_trigger_isolation(self):
    wf = self.workflow_create(self.wf_def)
    self.cron_trigger_create('trigger', wf[0]['Name'], '{}', '5 * * * *')
    alt_trs = self.mistral_alt_user('cron-trigger-list')
    self.assertNotIn('trigger', [t['Name'] for t in alt_trs])