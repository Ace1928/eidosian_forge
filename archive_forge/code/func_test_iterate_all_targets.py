from unittest import mock
from os_brick.initiator.connectors import base_iscsi
from os_brick.initiator.connectors import fake
from os_brick.tests import base as test_base
@mock.patch.object(base_iscsi.BaseISCSIConnector, '_get_all_targets')
def test_iterate_all_targets(self, mock_get_all_targets):
    connection_properties = {'target_portals': mock.sentinel.target_portals, 'target_iqns': mock.sentinel.target_iqns, 'target_luns': mock.sentinel.target_luns, 'extra_property': 'extra_property'}
    mock_get_all_targets.return_value = [(mock.sentinel.portal, mock.sentinel.iqn, mock.sentinel.lun)]
    list_props = list(self.connector._iterate_all_targets(connection_properties))
    mock_get_all_targets.assert_called_once_with(connection_properties)
    self.assertEqual(1, len(list_props))
    expected_props = {'target_portal': mock.sentinel.portal, 'target_iqn': mock.sentinel.iqn, 'target_lun': mock.sentinel.lun, 'extra_property': 'extra_property'}
    self.assertEqual(expected_props, list_props[0])