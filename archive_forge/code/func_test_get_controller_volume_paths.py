from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_controller_volume_paths(self):
    self._prepare_mock_disk()
    mock_disks = {self._FAKE_RES_PATH: self._FAKE_HOST_RESOURCE}
    disks = self._vmutils.get_controller_volume_paths(self._FAKE_RES_PATH)
    self.assertEqual(mock_disks, disks)