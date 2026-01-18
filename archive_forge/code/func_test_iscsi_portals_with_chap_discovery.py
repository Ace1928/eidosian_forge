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
@mock.patch.object(iscsi.ISCSIConnector, '_run_iscsiadm_update_discoverydb')
@mock.patch.object(os.path, 'exists', return_value=True)
def test_iscsi_portals_with_chap_discovery(self, exists, update_discoverydb):
    location = '10.0.2.15:3260'
    name = 'volume-00000001'
    iqn = 'iqn.2010-10.org.openstack:%s' % name
    vol = {'id': 1, 'name': name}
    auth_method = 'CHAP'
    auth_username = 'fake_chap_username'
    auth_password = 'fake_chap_password'
    discovery_auth_method = 'CHAP'
    discovery_auth_username = 'fake_chap_username'
    discovery_auth_password = 'fake_chap_password'
    connection_properties = self.iscsi_connection_chap(vol, location, iqn, auth_method, auth_username, auth_password, discovery_auth_method, discovery_auth_username, discovery_auth_password)
    self.connector_with_multipath = iscsi.ISCSIConnector(None, execute=self.fake_execute, use_multipath=True)
    self.cmds = []
    update_discoverydb.side_effect = [putils.ProcessExecutionError(None, None, 6), ('', ''), putils.ProcessExecutionError(None, None, 9)]
    self.connector_with_multipath._discover_iscsi_portals(connection_properties['data'])
    update_discoverydb.assert_called_with(connection_properties['data'])
    expected_cmds = ['iscsiadm -m discoverydb -t sendtargets -p %s -I default --op new' % location, 'iscsiadm -m node --op show -p %s' % location, 'iscsiadm -m discoverydb -t sendtargets -I default -p %s --discover' % location, 'iscsiadm -m node --op show -p %s' % location]
    self.assertEqual(expected_cmds, self.cmds)
    self.assertRaises(exception.TargetPortalNotFound, self.connector_with_multipath.connect_volume, connection_properties['data'])