from unittest import mock
import testscenarios
from testscenarios import scenarios as scnrs
import testtools
from heatclient.v1 import stacks
def test_files_show(self):
    manager = mock.MagicMock()
    stack = mock_stack(manager, 'files_stack', 'files1')
    stack.files()
    manager.files.assert_called_once_with('files_stack/files1')