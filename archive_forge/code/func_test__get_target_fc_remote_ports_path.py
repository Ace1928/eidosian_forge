import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch('builtins.open')
def test__get_target_fc_remote_ports_path(self, mock_open):
    path = '/sys/class/fc_remote_ports/rport-6:'
    execute_results = [('/sys/class/fc_remote_ports/rport-6:0-1/port_name\n', ''), ('1\n', '')]
    scsi_target_path = '/sys/class/fc_remote_ports/rport-6:0-1/scsi_target_id'
    mock_open.return_value.__enter__.return_value.read.return_value = '1\n'
    hbas, con_props = self.__get_rescan_info()
    with mock.patch.object(self.lfc, '_execute', side_effect=execute_results) as execute_mock:
        ctl = self.lfc._get_target_fc_remote_ports_path(path, con_props['target_wwn'][0], 1)
        expected_cmds = [mock.call('grep -Gil "514f0c50023f6c00" /sys/class/fc_remote_ports/rport-6:*/port_name', shell=True)]
        execute_mock.assert_has_calls(expected_cmds)
        mock_open.assert_called_once_with(scsi_target_path)
    self.assertEqual(['0', '1', 1], ctl)