import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_utils.FCUtils, '_wwn_struct_from_hex_str')
@mock.patch.object(fc_utils.FCUtils, '_open_adapter_by_wwn')
@mock.patch.object(fc_utils.FCUtils, '_close_adapter')
@mock.patch.object(fc_utils.FCUtils, '_get_target_mapping')
def test_get_fc_target_mapping(self, mock_get_target_mapping, mock_close_adapter, mock_open_adapter, mock_wwn_struct_from_hex_str):
    fake_node_wwn = list(range(8))
    fake_port_wwn = list(range(8)[::-1])
    mock_fcp_mappings = mock.MagicMock()
    mock_entry = mock.MagicMock()
    mock_entry.FcpId.NodeWWN.wwn = fake_node_wwn
    mock_entry.FcpId.PortWWN.wwn = fake_port_wwn
    mock_fcp_mappings.Entries = [mock_entry]
    mock_get_target_mapping.return_value = mock_fcp_mappings
    resulted_mappings = self._fc_utils.get_fc_target_mappings(mock.sentinel.local_wwnn)
    expected_mappings = [{'node_name': _utils.byte_array_to_hex_str(fake_node_wwn), 'port_name': _utils.byte_array_to_hex_str(fake_port_wwn), 'device_name': mock_entry.ScsiId.OSDeviceName, 'lun': mock_entry.ScsiId.ScsiOSLun, 'fcp_lun': mock_entry.FcpId.FcpLun}]
    self.assertEqual(expected_mappings, resulted_mappings)
    mock_wwn_struct_from_hex_str.assert_called_once_with(mock.sentinel.local_wwnn)
    mock_open_adapter.assert_called_once_with(mock_wwn_struct_from_hex_str.return_value)
    mock_close_adapter.assert_called_once_with(mock_open_adapter.return_value)