import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_get_published(self):
    self.client.executions.get.return_value = EXEC_WITH_PUBLISHED
    self.call(execution_cmd.GetPublished, app_args=['id'])
    self.assertDictEqual(EXEC_PUBLISHED, jsonutils.loads(self.app.stdout.write.call_args[0][0]))