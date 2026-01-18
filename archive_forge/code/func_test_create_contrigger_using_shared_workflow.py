from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_create_contrigger_using_shared_workflow(self):
    self._update_shared_workflow(new_status='accepted')
    trigger = self.cron_trigger_create('test_trigger', self.wf[0]['ID'], '{}', '5 * * * *', admin=False)
    wf_name = self.get_field_value(trigger, 'Workflow')
    self.assertEqual(self.wf[0]['Name'], wf_name)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'workflow-delete', params=self.wf[0]['ID'])