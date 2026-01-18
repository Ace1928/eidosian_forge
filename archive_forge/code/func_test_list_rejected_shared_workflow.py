from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_list_rejected_shared_workflow(self):
    self._update_shared_workflow(new_status='rejected')
    alt_wfs = self.mistral_alt_user('workflow-list')
    self.assertNotIn(self.wf[0]['ID'], [w['ID'] for w in alt_wfs])