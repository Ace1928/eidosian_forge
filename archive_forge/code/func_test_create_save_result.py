import copy
import io
import sys
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import action_executions as action_ex
from mistralclient.commands.v2 import action_executions as action_ex_cmd
from mistralclient.tests.unit import base
def test_create_save_result(self):
    self.client.action_executions.create.return_value = ACTION_EX_WITH_OUTPUT
    result = self.call(action_ex_cmd.Create, app_args=['some', '{"output": "Hello!"}', '--save-result'])
    self.assertEqual(('123', 'some', 'thing', '', 'task1', '1-2-3-4', 'RUNNING', 'RUNNING somehow.', True, '1', '1'), result[1])