import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_get_sub_wf_ex(self):
    self.client.executions.get.return_value = SUB_WF_EXEC
    result = self.call(execution_cmd.Get, app_args=['id'])
    self.assertEqual(SUB_WF_EX_RESULT, result[1])