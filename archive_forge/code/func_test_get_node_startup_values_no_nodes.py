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
def test_get_node_startup_values_no_nodes(self, exec_mock):
    connection_properties = {'target_portal': 'ip1:port1'}
    no_nodes_output = ''
    no_nodes_err = 'iscsiadm: No records found\n'
    exec_mock.return_value = (no_nodes_output, no_nodes_err)
    res = self.connector._get_node_startup_values(connection_properties)
    self.assertEqual({}, res)
    exec_mock.assert_called_once_with('iscsiadm', '-m', 'node', '--op', 'show', '-p', connection_properties['target_portal'], root_helper=self.connector._root_helper, run_as_root=True, check_exit_code=(0, 21))