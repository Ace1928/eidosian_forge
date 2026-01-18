import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_within_namespace_create_delete(self):
    namespace = 'abc'
    self.workflow_create(self.lowest_level_wf)
    self.workflow_create(self.lowest_level_wf, namespace=namespace)
    self.workflow_create(self.middle_wf, namespace=namespace)
    self.workflow_create(self.top_level_wf)
    wfs = self.workflow_create(self.top_level_wf, namespace=namespace)
    top_wf_name = wfs[0]['Name']
    execution = self.mistral_admin('execution-create', params='{0} --namespace {1}'.format(top_wf_name, namespace))
    exec_id = self.get_field_value(execution, 'ID')
    self.assertTableStruct(execution, ['Field', 'Value'])
    wf_name = self.get_field_value(execution, 'Workflow name')
    wf_namespace = self.get_field_value(execution, 'Workflow namespace')
    wf_id = self.get_field_value(execution, 'Workflow ID')
    created_at = self.get_field_value(execution, 'Created at')
    self.assertEqual(top_wf_name, wf_name)
    self.assertEqual(namespace, wf_namespace)
    self.assertIsNotNone(wf_id)
    self.assertIsNotNone(created_at)
    execs = self.mistral_admin('execution-list')
    self.assertIn(exec_id, [ex['ID'] for ex in execs])
    self.assertIn(wf_name, [ex['Workflow name'] for ex in execs])
    self.assertIn(namespace, [ex['Workflow namespace'] for ex in execs])
    params = '{} --force'.format(exec_id)
    self.mistral_admin('execution-delete', params=params)