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
@mock.patch.object(iscsi.ISCSIConnector, '_run_iscsi_session')
def test_get_iscsi_sessions_full(self, sessions_mock):
    iscsiadm_result = 'tcp: [session1] ip1:port1,1 tgt1 (non-flash)\ntcp: [session2] ip2:port2,-1 tgt2 (non-flash)\ntcp: [session3] ip3:port3,1 tgt3\n'
    sessions_mock.return_value = (iscsiadm_result, '')
    res = self.connector._get_iscsi_sessions_full()
    expected = [('tcp:', 'session1', 'ip1:port1', '1', 'tgt1'), ('tcp:', 'session2', 'ip2:port2', '-1', 'tgt2'), ('tcp:', 'session3', 'ip3:port3', '1', 'tgt3')]
    self.assertListEqual(expected, res)