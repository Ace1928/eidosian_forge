from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_detach_vm_disk(self):
    mock_disk = self._prepare_mock_disk()
    self._vmutils.detach_vm_disk(self._FAKE_VM_NAME, self._FAKE_HOST_RESOURCE, serial=mock.sentinel.serial)
    self._vmutils._jobutils.remove_virt_resource.assert_called_once_with(mock_disk)