from unittest import mock
from osc_lib import utils
from troveclient import common
from troveclient import exceptions
from troveclient.osc.v1 import database_configurations
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_configuration_delete_with_exception(self, mock_find):
    args = ['fakeconfig']
    parsed_args = self.check_parser(self.cmd, args, [])
    mock_find.side_effect = exceptions.CommandError
    self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)