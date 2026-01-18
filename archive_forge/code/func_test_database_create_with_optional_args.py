from unittest import mock
from osc_lib import exceptions
from osc_lib import utils
from troveclient import common
from troveclient.osc.v1 import databases
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_database_create_with_optional_args(self, mock_find):
    args = ['instance2', 'db2', '--character_set', 'utf8', '--collate', 'utf8_general_ci']
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, [])
    database_dict = {'name': 'db2', 'collate': 'utf8_general_ci', 'character_set': 'utf8'}
    result = self.cmd.take_action(parsed_args)
    self.database_client.create.assert_called_with('instance2', [database_dict])
    self.assertIsNone(result)