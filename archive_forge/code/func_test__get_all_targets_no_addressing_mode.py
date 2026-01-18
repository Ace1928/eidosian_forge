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
@ddt.data(None, 'SAM2')
@mock.patch.object(linuxscsi.LinuxSCSI, 'lun_for_addressing')
@mock.patch.object(iscsi.base_iscsi.BaseISCSIConnector, '_get_all_targets')
def test__get_all_targets_no_addressing_mode(self, addressing_mode, get_mock, luns_mock):
    get_mock.return_value = [(mock.sentinel.portal1, mock.sentinel.iqn1, mock.sentinel.lun1), (mock.sentinel.portal2, mock.sentinel.iqn2, mock.sentinel.lun2)]
    luns_mock.side_effect = [mock.sentinel.lun1B, mock.sentinel.lun2B]
    conn_props = self.CON_PROPS.copy()
    if addressing_mode:
        conn_props['addressing_mode'] = addressing_mode
    res = self.connector._get_all_targets(conn_props)
    self.assertEqual(2, luns_mock.call_count)
    luns_mock.assert_has_calls([mock.call(mock.sentinel.lun1, addressing_mode), mock.call(mock.sentinel.lun2, addressing_mode)])
    get_mock.assert_called_once_with(conn_props)
    expected = [(mock.sentinel.portal1, mock.sentinel.iqn1, mock.sentinel.lun1B), (mock.sentinel.portal2, mock.sentinel.iqn2, mock.sentinel.lun2B)]
    self.assertListEqual(expected, res)