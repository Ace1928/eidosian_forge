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
@mock.patch.object(iscsi.ISCSIConnector, '_cleanup_connection')
@mock.patch.object(iscsi.ISCSIConnector, '_connect_multipath_volume')
@mock.patch.object(iscsi.ISCSIConnector, '_connect_single_volume')
def test_connect_volume_mp(self, con_single_mock, con_mp_mock, clean_mock):
    self.connector.use_multipath = True
    res = self.connector.connect_volume(self.CON_PROPS)
    self.assertEqual(con_mp_mock.return_value, res)
    con_single_mock.assert_not_called()
    con_mp_mock.assert_called_once_with(self.CON_PROPS)
    clean_mock.assert_not_called()