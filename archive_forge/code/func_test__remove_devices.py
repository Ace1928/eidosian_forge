import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@ddt.data(('/dev/mapper/<WWN>', True), ('/dev/mapper/mpath0', True), ('/dev/sda', False), ('/dev/disk/by-path/pci-1-fc-1-lun-1', False))
@ddt.unpack
@mock.patch('os_brick.initiator.linuxscsi.LinuxSCSI.remove_scsi_device')
@mock.patch('os_brick.initiator.linuxscsi.LinuxSCSI.requires_flush')
@mock.patch('os_brick.utils.get_dev_path')
def test__remove_devices(self, path_used, was_multipath, get_dev_path_mock, flush_mock, remove_mock):
    exc = exception.ExceptionChainer()
    get_dev_path_mock.return_value = path_used
    self.connector._remove_devices(mock.sentinel.con_props, [{'device': '/dev/sda'}], mock.sentinel.device_info, force=False, exc=exc)
    self.assertFalse(bool(exc))
    get_dev_path_mock.assert_called_once_with(mock.sentinel.con_props, mock.sentinel.device_info)
    flush_mock.assert_called_once_with('/dev/sda', path_used, was_multipath)
    remove_mock.assert_called_once_with('/dev/sda', flush=flush_mock.return_value)