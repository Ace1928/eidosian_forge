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
@mock.patch.object(clusterutils.ClusterUtils, 'cluster_enum')
@mock.patch.object(clusterutils.ClusterUtils, 'get_cluster_group_type')
def test_get_vm_groups(self, mock_get_type, mock_cluster_enum):
    mock_groups = [mock.MagicMock(), mock.MagicMock(), mock.MagicMock()]
    group_types = [w_const.ClusGroupTypeVirtualMachine, w_const.ClusGroupTypeVirtualMachine, mock.sentinel.some_other_group_type]
    mock_cluster_enum.return_value = mock_groups
    mock_get_type.side_effect = group_types
    exp = mock_groups[:-1]
    res = list(self._clusterutils._get_vm_groups())
    self.assertEqual(exp, res)
    mock_cluster_enum.assert_called_once_with(w_const.CLUSTER_ENUM_GROUP)
    mock_get_type.assert_has_calls([mock.call(r['name']) for r in mock_groups])