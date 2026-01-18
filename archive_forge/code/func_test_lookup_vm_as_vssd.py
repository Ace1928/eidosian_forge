from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_lookup_vm_as_vssd(self):
    vssd = mock.MagicMock()
    expected_vssd = mock.MagicMock(VirtualSystemType=self._vmutils._VIRTUAL_SYSTEM_TYPE_REALIZED)
    self._vmutils._conn.Msvm_VirtualSystemSettingData.return_value = [vssd, expected_vssd]
    vssd = self._vmutils._lookup_vm_check(self._FAKE_VM_NAME)
    self.assertEqual(expected_vssd, vssd)