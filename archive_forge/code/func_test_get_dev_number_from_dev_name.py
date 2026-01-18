from unittest import mock
import ddt
from os_win import _utils
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage import diskutils
def test_get_dev_number_from_dev_name(self):
    fake_physical_device_name = '\\\\.\\PhysicalDrive15'
    expected_device_number = '15'
    get_dev_number = self._diskutils.get_device_number_from_device_name
    resulted_dev_number = get_dev_number(fake_physical_device_name)
    self.assertEqual(expected_device_number, resulted_dev_number)