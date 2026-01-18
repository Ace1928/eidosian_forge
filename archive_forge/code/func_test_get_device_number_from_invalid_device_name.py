from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_get_device_number_from_invalid_device_name(self):
    fake_physical_device_name = ''
    self.assertRaises(exceptions.DiskNotFound, self._diskutils.get_device_number_from_device_name, fake_physical_device_name)