from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
@mock.patch('os_win.utils.hostutils.kernel32')
def test_get_host_tick_count64(self, mock_kernel32):
    tick_count64 = '100'
    mock_kernel32.GetTickCount64.return_value = tick_count64
    response = self._hostutils.get_host_tick_count64()
    self.assertEqual(tick_count64, response)