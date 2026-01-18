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
@mock.patch.object(linuxscsi.LinuxSCSI, 'find_sysfs_multipath_dm', side_effect=[None, 'dm-0'])
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwn', return_value='wwn')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_add_path')
@mock.patch.object(linuxscsi.LinuxSCSI, 'multipath_add_wwid')
@mock.patch.object(iscsi.ISCSIConnector, '_connect_vol')
@mock.patch('os_brick.utils._time_sleep')
def test_connect_multipath_volume_some_fail_mp_found(self, sleep_mock, connect_mock, add_wwid_mock, add_path_mock, get_wwn_mock, find_dm_mock):

    def my_connect(rescans, props, data):
        devs = {'tgt1': '', 'tgt2': 'sdb', 'tgt3': '', 'tgt4': 'sdd'}
        data['stopped_threads'] += 1
        dev = devs[props['target_iqn']]
        if dev:
            data['num_logins'] += 1
            data['found_devices'].append(dev)
            data['just_added_devices'].append(dev)
        else:
            data['failed_logins'] += 1
    connect_mock.side_effect = my_connect
    res = self.connector._connect_multipath_volume(self.CON_PROPS)
    expected = {'type': 'block', 'scsi_wwn': 'wwn', 'multipath_id': 'wwn', 'path': '/dev/dm-0'}
    self.assertEqual(expected, res)
    self.assertEqual(1, get_wwn_mock.call_count)
    result = list(get_wwn_mock.call_args[0][0])
    result.sort()
    self.assertEqual(['sdb', 'sdd'], result)
    add_wwid_mock.assert_called_once_with('wwn')
    self.assertNotEqual(0, add_path_mock.call_count)
    self.assertGreaterEqual(find_dm_mock.call_count, 2)
    self.assertEqual(4, connect_mock.call_count)