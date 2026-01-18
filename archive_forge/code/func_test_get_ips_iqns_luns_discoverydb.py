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
@mock.patch.object(iscsi.ISCSIConnector, '_get_discoverydb_portals')
@mock.patch.object(iscsi.ISCSIConnector, '_discover_iscsi_portals')
def test_get_ips_iqns_luns_discoverydb(self, discover_mock, db_portals_mock):
    db_portals_mock.return_value = [('ip1:port1', 'tgt1', '1'), ('ip2:port2', 'tgt2', '2')]
    res = self.connector._get_ips_iqns_luns(self.SINGLE_CON_PROPS, discover=False)
    self.assertListEqual(db_portals_mock.return_value, res)
    db_portals_mock.assert_called_once_with(self.SINGLE_CON_PROPS)
    discover_mock.assert_not_called()