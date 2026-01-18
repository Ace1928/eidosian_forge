from unittest import mock
from unittest.mock import patch
import uuid
import testtools
from troveclient.v1 import backups
def test_execution_list(self):
    instance = mock.Mock(id='the_uuid')
    wf_input = '{"name": "wf2", "instance": "%s"}' % instance.id
    wf_name = self.backups.backup_create_workflow
    execution_list_result = [[mock.Mock(id=1, input=wf_input, workflow_name=wf_name, to_dict=mock.Mock(return_value={'id': 1})), mock.Mock(id=2, input='{}', workflow_name=wf_name)], [mock.Mock(id=3, input=wf_input, workflow_name=wf_name, to_dict=mock.Mock(return_value={'id': 3})), mock.Mock(id=4, input='{}', workflow_name=wf_name)], [mock.Mock(id=5, input=wf_input, workflow_name=wf_name, to_dict=mock.Mock(return_value={'id': 5})), mock.Mock(id=6, input='{}', workflow_name=wf_name)], [mock.Mock(id=7, input=wf_input, workflow_name='bar'), mock.Mock(id=8, input='{}', workflow_name=wf_name)]]
    cron_triggers = mock.Mock()
    cron_triggers.get = mock.Mock(return_value=mock.Mock(workflow_name=wf_name, workflow_input=wf_input))
    mistral_executions = mock.Mock()
    mistral_executions.list = mock.Mock(side_effect=execution_list_result)
    mistral_client = mock.Mock(cron_triggers=cron_triggers, executions=mistral_executions)
    el = self.backups.execution_list('dummy', mistral_client, limit=2)
    self.assertEqual(2, len(el))
    el = self.backups.execution_list('dummy', mistral_client, limit=2)
    self.assertEqual(1, len(el))
    the_exec = el.pop()
    self.assertEqual(5, the_exec.id)