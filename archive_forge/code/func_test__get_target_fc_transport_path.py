import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
def test__get_target_fc_transport_path(self):
    path = '/sys/class/fc_transport/target6:'
    execute_results = ('/sys/class/fc_transport/target6:0:1/port_name\n', '')
    _, con_props = self.__get_rescan_info()
    with mock.patch.object(self.lfc, '_execute', return_value=execute_results) as execute_mock:
        ctl = self.lfc._get_target_fc_transport_path(path, con_props['target_wwn'][0], 1)
        execute_mock.assert_called_once_with('grep -Gil "514f0c50023f6c00" /sys/class/fc_transport/target6:*/port_name', shell=True)
    self.assertEqual(['0', '1', 1], ctl)