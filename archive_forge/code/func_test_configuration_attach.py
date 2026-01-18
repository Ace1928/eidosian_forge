from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_configuration_attach(self, mock_find):
    args = ['instance1', 'config1']
    mock_find.side_effect = ['instance1', 'config1']
    parsed_args = self.check_parser(self.cmd, args, [])
    result = self.cmd.take_action(parsed_args)
    self.instance_client.update.assert_called_with('instance1', configuration='config1')
    self.assertIsNone(result)