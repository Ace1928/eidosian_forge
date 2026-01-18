import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_act_execution_list_with_limit(self):
    self.wait_execution_success(self.direct_ex_id)
    act_execs = self.mistral_admin('action-execution-list')
    self.assertGreater(len(act_execs), 1)
    act_execs = self.mistral_admin('action-execution-list', params='--limit 1')
    self.assertEqual(len(act_execs), 1)
    act_ex = act_execs[0]
    self.assertEqual(self.direct_wf['Name'], act_ex['Workflow name'])
    self.assertEqual('SUCCESS', act_ex['State'])