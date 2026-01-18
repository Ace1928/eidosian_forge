from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_serial_port_conns(self):
    self._lookup_vm()
    mock_com_1 = mock.Mock()
    mock_com_1.Connection = []
    mock_com_2 = mock.Mock()
    mock_com_2.Connection = [mock.sentinel.pipe_path]
    self._vmutils._get_vm_serial_ports = mock.Mock(return_value=[mock_com_1, mock_com_2])
    ret_val = self._vmutils.get_vm_serial_port_connections(mock.sentinel.vm_name)
    expected_ret_val = [mock.sentinel.pipe_path]
    self.assertEqual(expected_ret_val, ret_val)