from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_attach_volume_to_controller_with_disk_serial(self):
    self._test_attach_volume_to_controller(disk_serial=mock.sentinel.serial)