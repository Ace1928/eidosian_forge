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
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_sessions_full', return_value=[])
def test_get_iscsi_sessions_no_sessions(self, sessions_mock):
    res = self.connector._get_iscsi_sessions()
    self.assertListEqual([], res)
    sessions_mock.assert_called()