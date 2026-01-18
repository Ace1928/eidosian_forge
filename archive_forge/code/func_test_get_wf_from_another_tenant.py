from tempest.lib import exceptions
from mistralclient.tests.functional.cli.v2 import base_v2
def test_get_wf_from_another_tenant(self):
    wf = self.workflow_create(self.wf_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_alt_user, 'workflow-get', params=wf[0]['ID'])