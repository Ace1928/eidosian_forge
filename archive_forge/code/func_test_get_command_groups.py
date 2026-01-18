import testscenarios
from unittest import mock
from cliff import command
from cliff import commandmanager
from cliff.tests import base
from cliff.tests import utils
def test_get_command_groups(self):
    mgr = FakeCommandManager('test')
    mock_cmd_one = mock.Mock()
    mgr.add_command('mock', mock_cmd_one)
    cmd_mock, name, args = mgr.find_command(['mock'])
    self.assertEqual(mock_cmd_one, cmd_mock)
    mgr.add_command_group('greek')
    gl = mgr.get_command_groups()
    self.assertEqual(['test', 'greek'], gl)