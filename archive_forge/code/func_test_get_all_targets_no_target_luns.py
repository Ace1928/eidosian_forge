from unittest import mock
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator.connectors import fake
from os_brick.tests import base as test_base
def test_get_all_targets_no_target_luns(self):
    portals = [mock.sentinel.portals1, mock.sentinel.portals2]
    iqns = [mock.sentinel.iqns1, mock.sentinel.iqns2]
    lun = mock.sentinel.luns
    connection_properties = {'target_portals': portals, 'target_iqns': iqns, 'target_lun': lun}
    all_targets = self.connector._get_all_targets(connection_properties)
    expected_targets = zip(portals, iqns, [lun, lun])
    self.assertEqual(list(expected_targets), list(all_targets))