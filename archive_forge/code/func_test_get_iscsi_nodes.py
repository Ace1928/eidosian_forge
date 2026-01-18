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
def test_get_iscsi_nodes(self, exec_mock):
    iscsiadm_result = 'ip1:port1,1 tgt1\nip2:port2,-1 tgt2\nip3:port3,1 tgt3\n'
    exec_mock.return_value = (iscsiadm_result, '')
    res = self.connector._get_iscsi_nodes()
    expected = [('ip1:port1', 'tgt1'), ('ip2:port2', 'tgt2'), ('ip3:port3', 'tgt3')]
    self.assertListEqual(expected, res)
    exec_mock.assert_called_once_with('iscsiadm', '-m', 'node', run_as_root=True, root_helper=self.connector._root_helper, check_exit_code=False)