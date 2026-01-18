from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import iscsi
from os_brick.tests.windows import test_base
@mock.patch.object(iscsi.WindowsISCSIConnector, '_get_all_paths')
def test_connect_volume_exc(self, mock_get_all_paths):
    fake_paths = [(mock.sentinel.initiator_name, mock.sentinel.target_portal, mock.sentinel.target_iqn, mock.sentinel.target_lun)] * 3
    mock_get_all_paths.return_value = fake_paths
    self._iscsi_utils.login_storage_target.side_effect = os_win_exc.OSWinException
    self._connector.use_multipath = True
    self.assertRaises(exception.BrickException, self._connector.connect_volume, connection_properties={})