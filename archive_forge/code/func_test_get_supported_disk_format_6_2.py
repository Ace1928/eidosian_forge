from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def test_get_supported_disk_format_6_2(self):
    self._tgutils._win_gteq_6_3 = False
    fmt = self._tgutils.get_supported_disk_format()
    self.assertEqual(constants.DISK_FORMAT_VHD, fmt)