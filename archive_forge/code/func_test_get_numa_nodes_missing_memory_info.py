from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
@mock.patch.object(hostutils.HostUtils, '_get_numa_memory_info')
def test_get_numa_nodes_missing_memory_info(self, mock_get_memory_info):
    mock_get_memory_info.return_value = None
    self._check_get_numa_nodes_missing_info()