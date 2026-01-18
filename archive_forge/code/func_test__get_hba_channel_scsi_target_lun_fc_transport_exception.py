import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test__get_hba_channel_scsi_target_lun_fc_transport_exception(self):
    execute_effects = [('/sys/class/fc_transport/target6:0:1/port_name\n', ''), Exception()]
    _, expected_cmds = self._get_expected_info()
    hbas, con_props = self.__get_rescan_info()
    with mock.patch.object(self.lfc, '_execute', side_effect=execute_effects) as execute_mock:
        res = self.lfc._get_hba_channel_scsi_target_lun(hbas[0], con_props)
        execute_mock.assert_has_calls(expected_cmds)
    expected = ([['0', '1', 1]], {1})
    self.assertEqual(expected, res)