import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_struct, 'get_target_mapping_struct')
def test_get_target_mapping(self, mock_get_target_mapping):
    fake_entry_count = 10
    hresults = [fc_utils.HBA_STATUS_ERROR_MORE_DATA, fc_utils.HBA_STATUS_OK]
    mock_mapping = mock.Mock(NumberOfEntries=fake_entry_count)
    mock_get_target_mapping.return_value = mock_mapping
    self._mock_run.side_effect = hresults
    resulted_mapping = self._fc_utils._get_target_mapping(mock.sentinel.hba_handle)
    expected_calls = [mock.call(fc_utils.hbaapi.HBA_GetFcpTargetMapping, mock.sentinel.hba_handle, self._ctypes.byref(mock_mapping), ignored_error_codes=[fc_utils.HBA_STATUS_ERROR_MORE_DATA])] * 2
    self._mock_run.assert_has_calls(expected_calls)
    self.assertEqual(mock_mapping, resulted_mapping)
    mock_get_target_mapping.assert_has_calls([mock.call(0), mock.call(fake_entry_count)])