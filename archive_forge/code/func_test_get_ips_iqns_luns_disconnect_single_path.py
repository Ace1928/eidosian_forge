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
@mock.patch.object(iscsi.ISCSIConnector, '_get_all_targets')
@mock.patch.object(iscsi.ISCSIConnector, '_get_discoverydb_portals')
@mock.patch.object(iscsi.ISCSIConnector, '_discover_iscsi_portals')
def test_get_ips_iqns_luns_disconnect_single_path(self, discover_mock, db_portals_mock, get_targets_mock):
    db_portals_mock.side_effect = exception.TargetPortalsNotFound
    res = self.connector._get_ips_iqns_luns(self.SINGLE_CON_PROPS, discover=False, is_disconnect_call=True)
    db_portals_mock.assert_called_once_with(self.SINGLE_CON_PROPS)
    discover_mock.assert_not_called()
    get_targets_mock.assert_called_once_with(self.SINGLE_CON_PROPS)
    self.assertEqual(get_targets_mock.return_value, res)