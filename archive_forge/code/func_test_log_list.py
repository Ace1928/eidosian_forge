from unittest import mock
from osc_lib import utils
from troveclient.osc.v1 import database_logs
from troveclient.tests.osc.v1 import fakes
@mock.patch.object(utils, 'find_resource')
def test_log_list(self, mock_find):
    args = ['instance']
    mock_find.return_value = args[0]
    parsed_args = self.check_parser(self.cmd, args, [])
    columns, data = self.cmd.take_action(parsed_args)
    self.assertEqual(self.columns, columns)
    self.assertEqual(self.values, data)