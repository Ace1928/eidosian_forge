from unittest import mock
from mistralclient.api.v2 import actions
from mistralclient.commands.v2 import actions as action_cmd
from mistralclient.commands.v2 import base as cmd_base
from mistralclient.tests.unit import base
def test_get_definition(self):
    self.client.actions.get.return_value = ACTION_WITH_DEF
    self.call(action_cmd.GetDefinition, app_args=['name'])
    self.app.stdout.write.assert_called_with(ACTION_DEF)