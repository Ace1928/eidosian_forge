import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_ex_create_missed_input(self):
    self.create_file('empty')
    wf = self.workflow_create(self.wf_def)
    self.assertRaises(exceptions.CommandFailed, self.mistral_admin, 'execution-create empty', params=wf[1]['Name'])