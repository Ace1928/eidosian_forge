from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_list_accepted_shared_workflow(self):
    wfs = self.mistral_alt_user('workflow-list')
    self.assertNotIn(self.wf[0]['ID'], [w['ID'] for w in wfs])
    self._update_shared_workflow(new_status='accepted')
    alt_wfs = self.mistral_alt_user('workflow-list')
    self.assertIn(self.wf[0]['ID'], [w['ID'] for w in alt_wfs])
    self.assertIn(self.get_project_id('admin'), [w['Project ID'] for w in alt_wfs])