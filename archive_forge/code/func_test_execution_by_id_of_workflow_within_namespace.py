import time
from tempest.lib import exceptions
from mistralclient.tests.functional.cli import base
from mistralclient.tests.functional.cli.v2 import base_v2
def test_execution_by_id_of_workflow_within_namespace(self):
    namespace = 'abc'
    wfs = self.workflow_create(self.lowest_level_wf, namespace=namespace)
    wf_def_name = wfs[0]['Name']
    wf_id = wfs[0]['ID']
    execution = self.execution_create(wf_id)
    self.assertTableStruct(execution, ['Field', 'Value'])
    wf_name = self.get_field_value(execution, 'Workflow name')
    wf_namespace = self.get_field_value(execution, 'Workflow namespace')
    wf_id = self.get_field_value(execution, 'Workflow ID')
    self.assertEqual(wf_def_name, wf_name)
    self.assertEqual(namespace, wf_namespace)
    self.assertIsNotNone(wf_id)