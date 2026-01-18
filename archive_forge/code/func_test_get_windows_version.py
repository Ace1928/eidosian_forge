from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_windows_version(self):
    os = mock.MagicMock()
    os.Version = self._FAKE_VERSION_GOOD
    self._hostutils._conn_cimv2.Win32_OperatingSystem.return_value = [os]
    hostutils.HostUtils._windows_version = None
    self.assertEqual(self._FAKE_VERSION_GOOD, self._hostutils.get_windows_version())