import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch('os.path.realpath')
@mock.patch('glob.glob', return_value=['/sys/class/fc_host/host0', '/sys/class/fc_host/host2'])
@mock.patch('builtins.open', side_effect=IOError)
def test_get_fc_hbas_fail(self, mock_open, mock_glob, mock_path):
    hbas = self.lfc.get_fc_hbas()
    mock_glob.assert_called_once_with('/sys/class/fc_host/*')
    self.assertListEqual([], hbas)
    self.assertEqual(2, mock_open.call_count)
    mock_open.assert_has_calls((mock.call('/sys/class/fc_host/host0/port_name', 'rt'), mock.call('/sys/class/fc_host/host2/port_name', 'rt')))
    self.assertEqual(2, mock_path.call_count)
    mock_path.assert_has_calls((mock.call('/sys/class/fc_host/host0'), mock.call('/sys/class/fc_host/host2')))