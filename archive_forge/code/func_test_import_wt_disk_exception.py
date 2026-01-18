from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
def test_import_wt_disk_exception(self):
    mock_wt_disk_cls = self._tgutils._conn_wmi.WT_Disk
    mock_wt_disk_cls.ImportWTDisk.side_effect = test_base.FakeWMIExc
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.import_wt_disk, mock.sentinel.vhd_path, mock.sentinel.wtd_name)
    mock_wt_disk_cls.ImportWTDisk.assert_called_once_with(DevicePath=mock.sentinel.vhd_path, Description=mock.sentinel.wtd_name)