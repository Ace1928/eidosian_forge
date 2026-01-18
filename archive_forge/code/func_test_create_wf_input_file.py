import pkg_resources as pkg
from unittest import mock
from oslo_serialization import jsonutils
from mistralclient.api.v2 import executions
from mistralclient.commands.v2 import executions as execution_cmd
from mistralclient.tests.unit import base
def test_create_wf_input_file(self):
    self.client.executions.create.return_value = EXEC
    path = pkg.resource_filename('mistralclient', 'tests/unit/resources/ctx.json')
    result = self.call(execution_cmd.Create, app_args=['id', path])
    self.assertEqual(EX_RESULT, result[1])