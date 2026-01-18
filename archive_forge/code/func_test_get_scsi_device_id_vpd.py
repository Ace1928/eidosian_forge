import ctypes
from unittest import mock
import six
from os_win import _utils
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.storage.initiator import fc_utils
from os_win.utils.winapi.libs import hbaapi as fc_struct
@mock.patch.object(fc_utils.FCUtils, '_send_scsi_inquiry_v2')
def test_get_scsi_device_id_vpd(self, mock_send_scsi_inq):
    self._fc_utils._get_scsi_device_id_vpd(mock.sentinel.hba_handle, mock.sentinel.port_wwn, mock.sentinel.remote_port_wwn, mock.sentinel.fcp_lun)
    mock_send_scsi_inq.assert_called_once_with(mock.sentinel.hba_handle, mock.sentinel.port_wwn, mock.sentinel.remote_port_wwn, mock.sentinel.fcp_lun, 1, 131)