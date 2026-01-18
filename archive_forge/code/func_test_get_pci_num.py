import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
def test_get_pci_num(self):
    hba = {'device_path': '/sys/devices/pci0000:00/0000:00:03.0/0000:05:00.3/host2/fc_host/host2'}
    platform, pci_num = self.connector._get_pci_num(hba)
    self.assertEqual('0000:05:00.3', pci_num)
    self.assertIsNone(platform)
    hba = {'device_path': '/sys/devices/pci0000:00/0000:00:03.0/0000:05:00.3/0000:06:00.6/host2/fc_host/host2'}
    platform, pci_num = self.connector._get_pci_num(hba)
    self.assertEqual('0000:06:00.6', pci_num)
    self.assertIsNone(platform)
    hba = {'device_path': '/sys/devices/pci0000:20/0000:20:03.0/0000:21:00.2/net/ens2f2/ctlr_2/host3/fc_host/host3'}
    platform, pci_num = self.connector._get_pci_num(hba)
    self.assertEqual('0000:21:00.2', pci_num)
    self.assertIsNone(platform)