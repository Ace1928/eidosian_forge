from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_host')
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_disk')
def test_add_disk_to_target_exception(self, mock_get_wt_disk, mock_get_wt_host):
    mock_wt_disk = mock_get_wt_disk.return_value
    mock_wt_host = mock_get_wt_host.return_value
    mock_wt_host.AddWTDisk.side_effect = test_base.FakeWMIExc
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.add_disk_to_target, mock.sentinel.wtd_name, mock.sentinel.target_name)
    mock_get_wt_disk.assert_called_once_with(mock.sentinel.wtd_name)
    mock_get_wt_host.assert_called_once_with(mock.sentinel.target_name)
    mock_wt_host.AddWTDisk.assert_called_once_with(mock_wt_disk.WTD)