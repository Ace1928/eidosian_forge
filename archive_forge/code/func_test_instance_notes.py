from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_instance_notes(self):
    mock_vm_settings = self._lookup_vm()
    mock_vm_settings.Notes = self._get_fake_instance_notes()
    notes = self._vmutils._get_instance_notes(mock.sentinel.vm_name)
    self.assertEqual(notes[0], self._FAKE_VM_UUID)