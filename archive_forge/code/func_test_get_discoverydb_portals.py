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
@ddt.data(True, False)
@mock.patch.object(iscsi.ISCSIConnector, '_get_transport')
@mock.patch.object(iscsi.ISCSIConnector, '_run_iscsiadm_bare')
def test_get_discoverydb_portals(self, is_iser, iscsiadm_mock, transport_mock):
    params = {'iqn1': self.SINGLE_CON_PROPS['target_iqn'], 'iqn2': 'iqn.2004-04.com.qnap:ts-831x:iscsi.cinder-2017.9ef', 'addr': self.SINGLE_CON_PROPS['target_portal'].replace(':', ','), 'ip1': self.SINGLE_CON_PROPS['target_portal'], 'ip2': '192.168.1.3:3260', 'transport': 'iser' if is_iser else 'default', 'other_transport': 'default' if is_iser else 'iser'}
    iscsiadm_mock.return_value = ('SENDTARGETS:\nDiscoveryAddress: 192.168.1.33,3260\nDiscoveryAddress: %(addr)s\nTarget: %(iqn1)s\n\tPortal: %(ip2)s,1\n\t\tIface Name: %(transport)s\n\tPortal: %(ip1)s,1\n\t\tIface Name: %(transport)s\n\tPortal: %(ip1)s,1\n\t\tIface Name: %(other_transport)s\nTarget: %(iqn2)s\n\tPortal: %(ip2)s,1\n\t\tIface Name: %(transport)s\n\tPortal: %(ip1)s,1\n\t\tIface Name: %(transport)s\nDiscoveryAddress: 192.168.1.38,3260\niSNS:\nNo targets found.\nSTATIC:\nNo targets found.\nFIRMWARE:\nNo targets found.\n' % params, None)
    transport_mock.return_value = 'iser' if is_iser else 'non-iser'
    res = self.connector._get_discoverydb_portals(self.SINGLE_CON_PROPS)
    expected = [(params['ip2'], params['iqn1'], self.SINGLE_CON_PROPS['target_lun']), (params['ip1'], params['iqn1'], self.SINGLE_CON_PROPS['target_lun']), (params['ip2'], params['iqn2'], self.SINGLE_CON_PROPS['target_lun']), (params['ip1'], params['iqn2'], self.SINGLE_CON_PROPS['target_lun'])]
    self.assertListEqual(expected, res)
    iscsiadm_mock.assert_called_once_with(['-m', 'discoverydb', '-o', 'show', '-P', 1])
    transport_mock.assert_called_once_with()