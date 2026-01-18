import os.path
from unittest import mock
import ddt
from os_brick.initiator import linuxfc
from os_brick.tests import base
@mock.patch('os.path.realpath')
@mock.patch('glob.glob', return_value=['/sys/class/fc_host/host0', '/sys/class/fc_host/host2'])
@mock.patch('builtins.open')
def test_get_fc_hbas(self, mock_open, mock_glob, mock_path):
    mock_open.return_value.__enter__.return_value.read.side_effect = ['0x50014380242b9750\n', '0x50014380242b9751\n', 'Online', '0x50014380242b9752\n', '0x50014380242b9753\n', 'Online']
    pci_path = '/sys/devices/pci0000:20/0000:20:03.0/0000:21:00.'
    host0_pci = f'{pci_path}0/host0/fc_host/host0'
    host2_pci = f'{pci_path}1/host2/fc_host/host2'
    mock_path.side_effect = [host0_pci, host2_pci]
    hbas = self.lfc.get_fc_hbas()
    expected = [{'ClassDevice': 'host0', 'ClassDevicepath': host0_pci, 'port_name': '0x50014380242b9750', 'node_name': '0x50014380242b9751', 'port_state': 'Online'}, {'ClassDevice': 'host2', 'ClassDevicepath': host2_pci, 'port_name': '0x50014380242b9752', 'node_name': '0x50014380242b9753', 'port_state': 'Online'}]
    self.assertListEqual(expected, hbas)
    mock_glob.assert_called_once_with('/sys/class/fc_host/*')
    self.assertEqual(6, mock_open.call_count)
    mock_open.assert_has_calls((mock.call('/sys/class/fc_host/host0/port_name', 'rt'), mock.call('/sys/class/fc_host/host0/node_name', 'rt'), mock.call('/sys/class/fc_host/host0/port_state', 'rt'), mock.call('/sys/class/fc_host/host2/port_name', 'rt'), mock.call('/sys/class/fc_host/host2/node_name', 'rt'), mock.call('/sys/class/fc_host/host2/port_state', 'rt')), any_order=True)
    self.assertEqual(2, mock_path.call_count)
    mock_path.assert_has_calls((mock.call('/sys/class/fc_host/host0'), mock.call('/sys/class/fc_host/host2')))