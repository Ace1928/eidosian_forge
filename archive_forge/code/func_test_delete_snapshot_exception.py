from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.target import iscsi_target_utils as tg_utils
@mock.patch.object(tg_utils.ISCSITargetUtils, '_get_wt_snapshot')
def test_delete_snapshot_exception(self, mock_get_wt_snap):
    mock_wt_snap = mock_get_wt_snap.return_value
    mock_wt_snap.Delete_.side_effect = test_base.FakeWMIExc
    self.assertRaises(exceptions.ISCSITargetException, self._tgutils.delete_snapshot, mock.sentinel.snap_name)
    mock_get_wt_snap.assert_called_once_with(mock.sentinel.snap_name, fail_if_not_found=False)