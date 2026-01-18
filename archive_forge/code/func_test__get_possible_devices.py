import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@mock.patch.object(linuxscsi.LinuxSCSI, 'lun_for_addressing')
@mock.patch.object(fibre_channel.FibreChannelConnector, '_get_pci_num')
def test__get_possible_devices(self, pci_mock, lun_mock):
    pci_mock.side_effect = [(mock.sentinel.platform1, mock.sentinel.pci_num1), (mock.sentinel.platform2, mock.sentinel.pci_num2)]
    hbas = [mock.sentinel.hba1, mock.sentinel.hba2]
    lun_mock.side_effect = [mock.sentinel.lun1B, mock.sentinel.lun2B] * 2
    targets = [('wwn1', mock.sentinel.lun1), ('wwn2', mock.sentinel.lun2)]
    res = self.connector._get_possible_devices(hbas, targets, mock.sentinel.addressing_mode)
    self.assertEqual(2, pci_mock.call_count)
    pci_mock.assert_has_calls([mock.call(mock.sentinel.hba1), mock.call(mock.sentinel.hba2)])
    self.assertEqual(4, lun_mock.call_count)
    lun_mock.assert_has_calls([mock.call(mock.sentinel.lun1, mock.sentinel.addressing_mode), mock.call(mock.sentinel.lun2, mock.sentinel.addressing_mode)] * 2)
    expected = [(mock.sentinel.platform1, mock.sentinel.pci_num1, '0xwwn1', mock.sentinel.lun1B), (mock.sentinel.platform1, mock.sentinel.pci_num1, '0xwwn2', mock.sentinel.lun2B), (mock.sentinel.platform2, mock.sentinel.pci_num2, '0xwwn1', mock.sentinel.lun1B), (mock.sentinel.platform2, mock.sentinel.pci_num2, '0xwwn2', mock.sentinel.lun2B)]
    self.assertEqual(expected, res)