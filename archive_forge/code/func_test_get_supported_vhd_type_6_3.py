from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def test_get_supported_vhd_type_6_3(self):
    self._tgutils._win_gteq_6_3 = True
    vhd_type = self._tgutils.get_supported_vhd_type()
    self.assertEqual(constants.VHD_TYPE_DYNAMIC, vhd_type)