from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
@mock.patch.object(_wqlutils, 'get_element_associated_class')
def test_get_vm_scsi_controller(self, mock_get_element_associated_class):
    self._prepare_get_vm_controller(self._vmutils._SCSI_CTRL_RES_SUB_TYPE, mock_get_element_associated_class)
    path = self._vmutils.get_vm_scsi_controller(self._FAKE_VM_NAME)
    self.assertEqual(self._FAKE_RES_PATH, path)