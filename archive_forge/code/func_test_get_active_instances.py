from unittest import mock
import ddt
from six.moves import range  # noqa
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.compute import vmutils
def test_get_active_instances(self):
    fake_vm = mock.MagicMock()
    type(fake_vm).ElementName = mock.PropertyMock(side_effect=['active_vm', 'inactive_vm'])
    type(fake_vm).EnabledState = mock.PropertyMock(side_effect=[constants.HYPERV_VM_STATE_ENABLED, constants.HYPERV_VM_STATE_DISABLED])
    self._vmutils.list_instances = mock.MagicMock(return_value=[mock.sentinel.fake_vm_name] * 2)
    self._vmutils._lookup_vm = mock.MagicMock(side_effect=[fake_vm] * 2)
    active_instances = self._vmutils.get_active_instances()
    self.assertEqual(['active_vm'], active_instances)