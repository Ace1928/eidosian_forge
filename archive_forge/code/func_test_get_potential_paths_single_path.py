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
@mock.patch.object(iscsi.ISCSIConnector, '_get_iscsi_sessions')
@mock.patch.object(iscsi.ISCSIConnector, '_get_device_path')
def test_get_potential_paths_single_path(self, get_path_mock, get_sessions_mock):
    get_path_mock.side_effect = [['path1'], ['path2'], ['path3', 'path4']]
    get_sessions_mock.return_value = ['ip1:port1', 'ip2:port2', 'ip3:port3']
    self.connector.use_multipath = False
    res = self.connector._get_potential_volume_paths(self.CON_PROPS)
    self.assertEqual({'path1', 'path2', 'path3', 'path4'}, set(res))
    get_sessions_mock.assert_called_once_with()