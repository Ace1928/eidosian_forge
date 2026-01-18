import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_action_execution_update(self):
    wfs = self.workflow_create(self.wf_def)
    direct_wf_exec = self.execution_create(wfs[0]['Name'])
    direct_ex_id = self.get_field_value(direct_wf_exec, 'ID')
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'action-execution-update', params='%s ERROR' % direct_ex_id)