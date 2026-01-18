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
@mock.patch.object(iscsi.ISCSIConnector, '_get_transport', return_value='')
@mock.patch.object(iscsi.ISCSIConnector, '_run_iscsiadm_bare')
def test_get_discoverydb_portals_error(self, iscsiadm_mock, transport_mock):
    """DiscoveryAddress is not present."""
    iscsiadm_mock.return_value = ('SENDTARGETS:\nDiscoveryAddress: 192.168.1.33,3260\nDiscoveryAddress: 192.168.1.38,3260\niSNS:\nNo targets found.\nSTATIC:\nNo targets found.\nFIRMWARE:\nNo targets found.\n', None)
    self.assertRaises(exception.TargetPortalsNotFound, self.connector._get_discoverydb_portals, self.SINGLE_CON_PROPS)
    iscsiadm_mock.assert_called_once_with(['-m', 'discoverydb', '-o', 'show', '-P', 1])
    transport_mock.assert_not_called()