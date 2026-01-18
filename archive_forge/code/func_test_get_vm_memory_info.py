from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_vm_memory_info(self, mock_get_element_associated_class):
    vmsetting = self._lookup_vm()
    mock_s = mock.MagicMock(**self._FAKE_MEMORY_INFO)
    mock_get_element_associated_class.return_value = [mock_s]
    memory = self._vmutils.get_vm_memory_info(self._FAKE_VM_NAME)
    self.assertEqual(self._FAKE_MEMORY_INFO, memory)
    mock_get_element_associated_class.assert_called_once_with(self._vmutils._compat_conn, self._vmutils._MEMORY_SETTING_DATA_CLASS, element_instance_id=vmsetting.InstanceID)