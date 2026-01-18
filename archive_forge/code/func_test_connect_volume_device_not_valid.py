import os
from unittest import mock
import ddt
from os_brick import exception
from os_brick.initiator.connectors import base
from os_brick.initiator.connectors import fibre_channel
from os_brick.initiator import linuxfc
from os_brick.initiator import linuxscsi
from os_brick.tests.initiator import test_connector
@mock.patch('eventlet.greenthread.sleep', mock.Mock())
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_multipath_device')
@mock.patch.object(linuxscsi.LinuxSCSI, 'wait_for_rw')
@mock.patch.object(os.path, 'exists', return_value=True)
@mock.patch.object(os.path, 'realpath', return_value='/dev/sdb')
@mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_hbas')
@mock.patch.object(linuxfc.LinuxFibreChannel, 'get_fc_hbas_info')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_scsi_wwn')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_device_info')
@mock.patch.object(base.BaseLinuxConnector, 'check_valid_device')
def test_connect_volume_device_not_valid(self, check_valid_device_mock, get_device_info_mock, get_scsi_wwn_mock, get_fc_hbas_info_mock, get_fc_hbas_mock, realpath_mock, exists_mock, wait_for_rw_mock, find_mp_dev_mock):
    check_valid_device_mock.return_value = False
    self.assertRaises(exception.NoFibreChannelVolumeDeviceFound, self._test_connect_volume_multipath, get_device_info_mock, get_scsi_wwn_mock, get_fc_hbas_info_mock, get_fc_hbas_mock, realpath_mock, exists_mock, wait_for_rw_mock, find_mp_dev_mock, 'rw', True)