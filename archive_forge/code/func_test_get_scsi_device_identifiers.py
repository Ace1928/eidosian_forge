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
@mock.patch.object(fc_utils.FCUtils, '_get_scsi_device_id_vpd')
def test_get_scsi_device_identifiers(self, mock_get_scsi_dev_id_vpd, mock_close_adapter, mock_open_adapter, mock_wwn_struct_from_hex_str):
    mock_wwn_struct_from_hex_str.side_effect = (mock.sentinel.local_wwnn_struct, mock.sentinel.local_wwpn_struct, mock.sentinel.remote_wwpn_struct)
    self._diskutils._parse_scsi_page_83.return_value = mock.sentinel.identifiers
    identifiers = self._fc_utils.get_scsi_device_identifiers(mock.sentinel.local_wwnn, mock.sentinel.local_wwpn, mock.sentinel.remote_wwpn, mock.sentinel.fcp_lun, mock.sentinel.select_supp_ids)
    self.assertEqual(mock.sentinel.identifiers, identifiers)
    mock_wwn_struct_from_hex_str.assert_has_calls([mock.call(wwn) for wwn in (mock.sentinel.local_wwnn, mock.sentinel.local_wwpn, mock.sentinel.remote_wwpn)])
    mock_get_scsi_dev_id_vpd.assert_called_once_with(mock_open_adapter.return_value, mock.sentinel.local_wwpn_struct, mock.sentinel.remote_wwpn_struct, mock.sentinel.fcp_lun)
    self._diskutils._parse_scsi_page_83.assert_called_once_with(mock_get_scsi_dev_id_vpd.return_value, select_supported_identifiers=mock.sentinel.select_supp_ids)
    mock_open_adapter.assert_called_once_with(mock.sentinel.local_wwnn_struct)
    mock_close_adapter.assert_called_once_with(mock_open_adapter.return_value)