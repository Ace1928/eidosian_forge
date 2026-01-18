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
@mock.patch.object(linuxscsi.LinuxSCSI, 'get_sysfs_wwn', return_value='')
@mock.patch.object(iscsi.ISCSIConnector, '_connect_vol')
@mock.patch.object(iscsi.ISCSIConnector, '_cleanup_connection')
@mock.patch('os_brick.utils._time_sleep')
def test_connect_single_volume_no_wwn(self, sleep_mock, cleanup_mock, connect_mock, get_wwn_mock):

    def my_connect(rescans, props, data):
        data['found_devices'].append('sdz')
    connect_mock.side_effect = my_connect
    res = self.connector._connect_single_volume(self.CON_PROPS)
    expected = {'type': 'block', 'scsi_wwn': '', 'path': '/dev/sdz'}
    self.assertEqual(expected, res)
    get_wwn_mock.assert_has_calls([mock.call(['sdz'])] * 10)
    self.assertEqual(10, get_wwn_mock.call_count)
    sleep_mock.assert_has_calls([mock.call(1)] * 10)
    self.assertEqual(10, sleep_mock.call_count)
    cleanup_mock.assert_not_called()