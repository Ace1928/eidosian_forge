from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2.executions import Execution
from mistralclient.api.v2 import tasks
from mistralclient.commands.v2 import tasks as task_cmd
from mistralclient.tests.unit import base
def test_get_result(self):
    self.client.tasks.get.return_value = TASK_WITH_RESULT
    self.call(task_cmd.GetResult, app_args=['id'])
    self.assertDictEqual(TASK_RESULT, jsonutils.loads(self.app.stdout.write.call_args[0][0]))