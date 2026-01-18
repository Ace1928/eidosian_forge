from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_lookup_vm_none(self):
    self._vmutils._conn.Msvm_ComputerSystem.return_value = []
    self.assertRaises(exceptions.HyperVVMNotFoundException, self._vmutils._lookup_vm_check, self._FAKE_VM_NAME, as_vssd=False)