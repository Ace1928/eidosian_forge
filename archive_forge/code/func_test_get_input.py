import copy
import io
import sys
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import action_executions as action_ex
from mistralclient.commands.v2 import action_executions as action_ex_cmd
from mistralclient.tests.unit import base
def test_get_input(self):
    self.client.action_executions.get.return_value = ACTION_EX_WITH_INPUT
    self.call(action_ex_cmd.GetInput, app_args=['id'])
    self.assertDictEqual(ACTION_EX_INPUT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))