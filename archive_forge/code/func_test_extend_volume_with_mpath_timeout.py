import os
import os.path
import textwrap
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.initiator import linuxscsi
from os_brick.tests import base
@mock.patch('os_brick.utils.check_valid_device')
@mock.patch('time.sleep')
@mock.patch('time.time')
@mock.patch.object(linuxscsi.LinuxSCSI, '_multipath_resize_map')
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_multipath_device_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_scsi_wwn')
@mock.patch('os_brick.utils.get_device_size')
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_device_info')
def test_extend_volume_with_mpath_timeout(self, mock_device_info, mock_device_size, mock_scsi_wwn, mock_find_mpath_path, mock_mpath_resize_map, mock_currtime, mock_sleep, mock_valid_dev):
    """Test extending a volume where there is a multipath device fail."""
    mock_device_info.side_effect = [{'host': host, 'channel': '0', 'id': '0', 'lun': '1'} for host in ['0', '1']]
    mock_device_size.side_effect = [1024, 2048, 1024, 2048, 1024, 2048]
    wwn = '1234567890123456'
    mock_scsi_wwn.return_value = wwn
    mock_find_mpath_path.return_value = '/dev/mapper/dm-uuid-mpath-%s' % wwn
    mock_valid_dev.return_value = True
    fake_time = 0

    def get_fake_time():
        nonlocal fake_time
        fake_time += 10
        return fake_time
    mock_currtime.side_effect = get_fake_time
    mock_mpath_resize_map.side_effect = putils.ProcessExecutionError(stdout='timeout')
    self.assertRaises(putils.ProcessExecutionError, self.linuxscsi.extend_volume, ['/dev/fake1', '/dev/fake2'], use_multipath=True)