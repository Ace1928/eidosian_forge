import ctypes
from unittest import mock
import ddt
from six.moves import queue
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils.compute import clusterutils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi.libs import clusapi as clusapi_def
from os_win.utils.winapi import wintypes
@mock.patch.object(clusterutils.ClusterUtils, '_lookup_vm_group')
def test_lookup_vm_group_check_no_vm(self, mock_lookup_vm_group):
    mock_lookup_vm_group.return_value = None
    self.assertRaises(exceptions.HyperVVMNotFoundException, self._clusterutils._lookup_vm_group_check, self._FAKE_VM_NAME)