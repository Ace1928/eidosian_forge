from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_get_new_setting_data')
def test_create_nic(self, mock_get_new_virt_res):
    mock_vm = self._lookup_vm()
    mock_nic = mock_get_new_virt_res.return_value
    self._vmutils.create_nic(self._FAKE_VM_NAME, self._FAKE_RES_NAME, self._FAKE_ADDRESS)
    self._vmutils._jobutils.add_virt_resource.assert_called_once_with(mock_nic, mock_vm)