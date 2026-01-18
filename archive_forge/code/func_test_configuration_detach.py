from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_configuration_detach(self, mock_find):
    args = ['instance2']
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, [])
    result = self.cmd.take_action(parsed_args)
    self.instance_client.update.assert_called_with('instance2', remove_configuration=True)
    self.assertIsNone(result)