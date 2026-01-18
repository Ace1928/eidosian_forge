import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_load_commands_replace_underscores(self):
    testcmd = mock.Mock()
    testcmd.name = 'test_cmd'
    mock_get_group_all = mock.Mock(return_value=[testcmd])
    with mock.patch('stevedore.ExtensionManager', mock_get_group_all) as mock_manager:
        mgr = commandmanager.CommandManager('test', convert_underscores=True)
        mock_manager.assert_called_once_with('test')
        names = [n for n, v in mgr]
        self.assertEqual(['test cmd'], names)