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
def test_get_cluster_nodes(self, mock_cluster_enum):
    expected = mock_cluster_enum.return_value
    self.assertEqual(expected, self._clusterutils._get_cluster_nodes())
    mock_cluster_enum.assert_called_once_with(w_const.CLUSTER_ENUM_NODE)