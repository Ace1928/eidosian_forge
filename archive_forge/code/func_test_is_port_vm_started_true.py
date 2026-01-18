from unittest import mock
import ddt
from oslo_utils import units
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import _wqlutils
from os_win.utils.network import networkutils
def test_is_port_vm_started_true(self):
    self._test_is_port_vm_started(self.netutils._HYPERV_VM_STATE_ENABLED, True)