from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_check_min_windows_version_false(self):
    self._test_check_min_windows_version(self._FAKE_VERSION_BAD, False)