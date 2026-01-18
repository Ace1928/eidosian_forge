import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_act_execution_get(self):
    self.wait_execution_success(self.direct_ex_id)
    task = self.mistral_admin('task-list', params=self.direct_ex_id)[0]
    act_ex_from_list = self.mistral_admin('action-execution-list', params=task['ID'])[0]
    act_ex = self.mistral_admin('action-execution-get', params=act_ex_from_list['ID'])
    wf_name = self.get_field_value(act_ex, 'Workflow name')
    state = self.get_field_value(act_ex, 'State')
    self.assertEqual(act_ex_from_list['ID'], self.get_field_value(act_ex, 'ID'))
    self.assertEqual(self.direct_wf['Name'], wf_name)
    self.assertEqual('SUCCESS', state)