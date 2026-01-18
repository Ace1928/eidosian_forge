import builtins
import functools
import io
import time
from unittest import mock
import ddt
from os_brick import exception
from os_brick.tests import base
from os_brick import utils
@ddt.data(('1024', 1024), ('junk', None), ('2048\n', 2048))
@ddt.unpack
def test_get_device_size(self, cmd_out, expected):
    mock_execute = mock.Mock()
    mock_execute._execute.return_value = (cmd_out, None)
    device = '/dev/fake'
    ret_size = utils.get_device_size(mock_execute, device)
    self.assertEqual(expected, ret_size)
    mock_execute._execute.assert_called_once_with('blockdev', '--getsize64', device, run_as_root=True, root_helper=mock_execute._root_helper)