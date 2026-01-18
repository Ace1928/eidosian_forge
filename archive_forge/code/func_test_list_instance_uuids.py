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
@mock.patch.object(clusterutils.ClusterUtils, '_get_vm_groups')
def test_list_instance_uuids(self, mock_get_vm_groups):
    mock_get_vm_groups.return_value = [dict(id='uuid1'), dict(id='uuid2')]
    ret = self._clusterutils.list_instance_uuids()
    self.assertCountEqual(['uuid1', 'uuid2'], ret)