from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_create_public_workflow(self):
    wf = self.workflow_create(self.wf_def, scope='public')
    same_wf = self.mistral_alt_user('workflow-get', params=wf[0]['Name'])
    self.assertEqual(wf[0]['Name'], self.get_field_value(same_wf, 'Name'))