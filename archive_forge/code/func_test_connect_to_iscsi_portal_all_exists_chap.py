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
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_sessions_full')
def test_connect_to_iscsi_portal_all_exists_chap(self, get_sessions_mock):
    """Node and session already exists and we use chap authentication."""
    session = 'session2'
    get_sessions_mock.return_value = [('tcp:', session, 'ip1:port1', '-1', 'tgt1')]
    con_props = self.CON_PROPS.copy()
    con_props.update(auth_method='CHAP', auth_username='user', auth_password='pwd')
    utils.ISCSI_SUPPORTS_MANUAL_SCAN = None
    res = self.connector._connect_to_iscsi_portal(con_props)
    self.assertEqual((session, True), res)
    self.assertTrue(utils.ISCSI_SUPPORTS_MANUAL_SCAN)
    prefix = 'iscsiadm -m node -T tgt1 -p ip1:port1'
    expected_cmds = [prefix, prefix + ' --op update -n node.session.scan -v manual', prefix + ' --op update -n node.session.auth.authmethod -v CHAP', prefix + ' --op update -n node.session.auth.username -v user', prefix + ' --op update -n node.session.auth.password -v pwd']
    self.assertListEqual(expected_cmds, self.cmds)
    get_sessions_mock.assert_called_once_with()