from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_modify_virtual_system')
@mock.patch.object(vmutils.VMUtils, '_lookup_vm_check')
def test_enable_secure_boot(self, mock_lookup_vm_check, mock_modify_virtual_system):
    vs_data = mock_lookup_vm_check.return_value
    with mock.patch.object(self._vmutils, '_set_secure_boot') as mock_set_secure_boot:
        self._vmutils.enable_secure_boot(mock.sentinel.VM_NAME, mock.sentinel.certificate_required)
        mock_lookup_vm_check.assert_called_with(mock.sentinel.VM_NAME)
        mock_set_secure_boot.assert_called_once_with(vs_data, mock.sentinel.certificate_required)
        mock_modify_virtual_system.assert_called_once_with(vs_data)