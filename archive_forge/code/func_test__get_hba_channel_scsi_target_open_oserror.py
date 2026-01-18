import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch('builtins.open')
def test__get_hba_channel_scsi_target_open_oserror(self, mock_open):
    execute_effects, expected_cmds = self._get_expected_info(targets=2, remote_scan=True)
    mock_open = mock_open.return_value.__enter__.return_value
    mock_open.read.side_effect = ['1\n', OSError()]
    hbas, con_props = self.__get_rescan_info()
    with mock.patch.object(self.lfc, '_execute', side_effect=execute_effects) as execute_mock:
        res = self.lfc._get_hba_channel_scsi_target_lun(hbas[0], con_props)
        execute_mock.assert_has_calls(expected_cmds)
    expected = ([['0', '1', 1]], set())
    self.assertEqual(expected, res)