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
@mock.patch('os_brick.utils._time_sleep', mock.Mock())
@mock.patch.object(linuxscsi.LinuxSCSI, 'scan_iscsi')
@mock.patch.object(iscsi.ISCSIConnector, '_connect_to_iscsi_portal')
def test_connect_vol_stop_connecting(self, connect_mock, scan_mock):
    data = self._get_connect_vol_data()

    def device_name_by_hctl(session, hctl):
        data['stop_connecting'] = True
        return None
    lscsi = self.connector._linuxscsi
    hctl = [mock.sentinel.host, mock.sentinel.channel, mock.sentinel.target, mock.sentinel.lun]
    connect_mock.return_value = (mock.sentinel.session, False)
    with mock.patch.object(lscsi, 'get_hctl', return_value=hctl) as hctl_mock, mock.patch.object(lscsi, 'device_name_by_hctl', side_effect=device_name_by_hctl) as dev_name_mock:
        self.connector._connect_vol(3, self.CON_PROPS, data)
    expected = self._get_connect_vol_data()
    expected.update(num_logins=1, stopped_threads=1, stop_connecting=True)
    self.assertDictEqual(expected, data)
    hctl_mock.assert_called_once_with(mock.sentinel.session, self.CON_PROPS['target_lun'])
    scan_mock.assert_not_called()
    dev_name_mock.assert_called_once_with(mock.sentinel.session, hctl)