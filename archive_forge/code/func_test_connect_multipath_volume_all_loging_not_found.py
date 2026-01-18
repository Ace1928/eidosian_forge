import collections
import os
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from os_brick import exception
from os_brick.initiator.connectors import iscsi
from os_brick.initiator import linuxscsi
from os_brick.initiator import utils
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests.initiator import test_connector
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_sysfs_multipath_dm', return_value=None)
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwn', return_value='wwn')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_add_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_add_wwid')
@mock.patch.object(iscsi.time, 'time', side_effect=(0, 0, 11, 0))
@mock.patch.object(iscsi.ISCSIConnector, '_connect_vol')
@mock.patch('os_brick.utils._time_sleep', mock.Mock())
def test_connect_multipath_volume_all_loging_not_found(self, connect_mock, time_mock, add_wwid_mock, add_path_mock, get_wwn_mock, find_dm_mock):

    def my_connect(rescans, props, data):
        data['stopped_threads'] += 1
        data['num_logins'] += 1
    connect_mock.side_effect = my_connect
    self.assertRaises(exception.VolumeDeviceNotFound, self.connector._connect_multipath_volume, self.CON_PROPS)
    get_wwn_mock.assert_not_called()
    add_wwid_mock.assert_not_called()
    add_path_mock.assert_not_called()
    find_dm_mock.assert_not_called()
    self.assertEqual(12, connect_mock.call_count)