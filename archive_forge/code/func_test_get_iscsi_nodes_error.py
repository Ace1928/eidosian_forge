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
@mock.patch.object(iscsi.ISCSIConnector, '_execute')
def test_get_iscsi_nodes_error(self, exec_mock):
    exec_mock.return_value = (None, 'error')
    res = self.connector._get_iscsi_nodes()
    self.assertEqual([], res)