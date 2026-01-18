from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import iscsi
from os_brick.tests.windows import test_base
@mock.patch.object(iscsi.WindowsISCSIConnector, '_get_all_targets')
def test_get_all_paths(self, mock_get_all_targets):
    initiators = [mock.sentinel.initiator_0, mock.sentinel.initiator_1]
    all_targets = [(mock.sentinel.portal_0, mock.sentinel.target_0, mock.sentinel.lun_0), (mock.sentinel.portal_1, mock.sentinel.target_1, mock.sentinel.lun_1)]
    self._connector.initiator_list = initiators
    mock_get_all_targets.return_value = all_targets
    expected_paths = [(initiator_name, target_portal, target_iqn, target_lun) for target_portal, target_iqn, target_lun in all_targets for initiator_name in initiators]
    all_paths = self._connector._get_all_paths(mock.sentinel.conn_props)
    self.assertEqual(expected_paths, all_paths)
    mock_get_all_targets.assert_called_once_with(mock.sentinel.conn_props)