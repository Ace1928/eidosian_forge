import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_act_execution_get_list_within_namespace(self):
    namespace = 'bbb'
    self.workflow_create(self.wf_def, namespace=namespace)
    wf_ex = self.execution_create(self.direct_wf['Name'] + ' --namespace ' + namespace)
    exec_id = self.get_field_value(wf_ex, 'ID')
    self.wait_execution_success(exec_id)
    task = self.mistral_admin('task-list', params=exec_id)[0]
    act_ex_from_list = self.mistral_admin('action-execution-list', params=task['ID'])[0]
    act_ex = self.mistral_admin('action-execution-get', params=act_ex_from_list['ID'])
    wf_name = self.get_field_value(act_ex, 'Workflow name')
    wf_namespace = self.get_field_value(act_ex, 'Workflow namespace')
    status = self.get_field_value(act_ex, 'State')
    self.assertEqual(act_ex_from_list['ID'], self.get_field_value(act_ex, 'ID'))
    self.assertEqual(self.direct_wf['Name'], wf_name)
    self.assertEqual('SUCCESS', status)
    self.assertEqual(namespace, wf_namespace)
    self.assertEqual(namespace, act_ex_from_list['Workflow namespace'])