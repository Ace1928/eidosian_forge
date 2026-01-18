from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_snapshot')
def test_export_snapshot_exception(self, mock_get_wt_snap):
    mock_wt_disk_cls = self._tgutils._conn_wmi.WT_Disk
    mock_wt_disk = mock.Mock()
    mock_wt_disk_cls.return_value = [mock_wt_disk]
    mock_wt_disk.Delete_.side_effect = test_base.FakeWMIExc
    mock_wt_snap = mock_get_wt_snap.return_value
    mock_wt_snap.Export.return_value = [mock.sentinel.wt_disk_id]
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.export_snapshot, mock.sentinel.snap_name, mock.sentinel.dest_path)
    mock_get_wt_snap.assert_called_once_with(mock.sentinel.snap_name)
    mock_wt_snap.Export.assert_called_once_with()
    mock_wt_disk_cls.assert_called_once_with(WTD=mock.sentinel.wt_disk_id)
    expected_wt_disk_description = '%s-%s-temp' % (mock.sentinel.snap_name, mock.sentinel.wt_disk_id)
    self.assertEqual(expected_wt_disk_description, mock_wt_disk.Description)
    mock_wt_disk.put.assert_called_once_with()
    mock_wt_disk.Delete_.assert_called_once_with()
    self._tgutils._pathutils.copy.assert_called_once_with(mock_wt_disk.DevicePath, mock.sentinel.dest_path)