from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import fibre_channel as fc
from os_brick.tests.windows import test_base
@mock.patch.object(fc.WindowsFCConnector, '_get_dev_nums_by_scsi_id')
def test_get_disk_paths_by_scsi_id(self, mock_get_dev_nums):
    remote_wwpns = [mock.sentinel.remote_wwpn_0, mock.sentinel.remote_wwpn_1]
    fake_init_target_map = {mock.sentinel.local_wwpn: remote_wwpns}
    conn_props = dict(initiator_target_map=fake_init_target_map)
    mock_get_dev_nums.side_effect = [os_win_exc.FCException, [mock.sentinel.dev_num]]
    mock_get_dev_name = self._diskutils.get_device_name_by_device_number
    mock_get_dev_name.return_value = mock.sentinel.dev_name
    disk_paths = self._connector._get_disk_paths_by_scsi_id(conn_props, mock.sentinel.fcp_lun)
    self.assertEqual([mock.sentinel.dev_name], disk_paths)
    mock_get_dev_nums.assert_has_calls([mock.call(mock.sentinel.local_wwpn, remote_wwpn, mock.sentinel.fcp_lun) for remote_wwpn in remote_wwpns])
    mock_get_dev_name.assert_called_once_with(mock.sentinel.dev_num)