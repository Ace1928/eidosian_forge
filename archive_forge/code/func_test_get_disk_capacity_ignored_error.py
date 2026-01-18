from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_get_disk_capacity_ignored_error(self):
    self._test_get_disk_capacity(raised_exc=exceptions.Win32Exception, ignore_errors=True)