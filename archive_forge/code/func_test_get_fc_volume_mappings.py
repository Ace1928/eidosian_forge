from unittest import mock
import ddt
from os_win import exceptions as os_win_exc
from os_brick import exception
from os_brick.initiator.windows import fibre_channel as fc
from os_brick.tests.windows import test_base
@mock.patch.object(fc.WindowsFCConnector, '_get_fc_hba_mappings')
def test_get_fc_volume_mappings(self, mock_get_fc_hba_mappings):
    fake_target_wwpn = 'FAKE_TARGET_WWPN'
    fake_conn_props = dict(target_lun=mock.sentinel.target_lun, target_wwn=[fake_target_wwpn])
    mock_hba_mappings = {mock.sentinel.node_name: mock.sentinel.hba_ports}
    mock_get_fc_hba_mappings.return_value = mock_hba_mappings
    all_target_mappings = [{'device_name': mock.sentinel.dev_name, 'port_name': fake_target_wwpn, 'lun': mock.sentinel.target_lun}, {'device_name': mock.sentinel.dev_name_1, 'port_name': mock.sentinel.target_port_name_1, 'lun': mock.sentinel.target_lun}, {'device_name': mock.sentinel.dev_name, 'port_name': mock.sentinel.target_port_name, 'lun': mock.sentinel.target_lun_1}]
    expected_mappings = [all_target_mappings[0]]
    self._fc_utils.get_fc_target_mappings.return_value = all_target_mappings
    volume_mappings = self._connector._get_fc_volume_mappings(fake_conn_props)
    self.assertEqual(expected_mappings, volume_mappings)