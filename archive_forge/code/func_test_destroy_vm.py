from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_destroy_vm(self):
    self._lookup_vm()
    mock_svc = self._vmutils._vs_man_svc
    getattr(mock_svc, self._DESTROY_SYSTEM).return_value = (self._FAKE_JOB_PATH, self._FAKE_RET_VAL)
    self._vmutils.destroy_vm(self._FAKE_VM_NAME)
    getattr(mock_svc, self._DESTROY_SYSTEM).assert_called_with(self._FAKE_VM_PATH)