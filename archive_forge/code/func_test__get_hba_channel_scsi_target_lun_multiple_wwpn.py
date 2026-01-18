import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch('builtins.open')
@ddt.data(True, False)
def test__get_hba_channel_scsi_target_lun_multiple_wwpn(self, remote_scan, mock_open):
    execute_results, expected_cmds = self._get_expected_info(targets=2)
    if remote_scan:
        mock_open = mock_open.return_value.__enter__.return_value
        mock_open.read.return_value = '1\n'
    hbas, con_props = self.__get_rescan_info()
    with mock.patch.object(self.lfc, '_execute', side_effect=execute_results) as execute_mock:
        res = self.lfc._get_hba_channel_scsi_target_lun(hbas[0], con_props)
        execute_mock.assert_has_calls(expected_cmds)
    expected = ([['0', '1', 1], ['0', '2', 1]], set())
    self.assertEqual(expected, res)