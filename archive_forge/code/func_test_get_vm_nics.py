from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(vmutils.VMUtils, '_lookup_vm_check')
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_vm_nics(self, mock_get_assoc, mock_lookup_vm):
    vnics = self._vmutils._get_vm_nics(mock.sentinel.vm_name)
    self.assertEqual(mock_get_assoc.return_value, vnics)
    mock_lookup_vm.assert_called_once_with(mock.sentinel.vm_name)
    mock_get_assoc.assert_called_once_with(self._vmutils._compat_conn, self._vmutils._SYNTHETIC_ETHERNET_PORT_SETTING_DATA_CLASS, element_instance_id=mock_lookup_vm.return_value.InstanceId)