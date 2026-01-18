from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_delete_wf_from_another_tenant(self):
    wf = self.workflow_create(self.wf_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workflow-delete', params=wf[0]['ID'])