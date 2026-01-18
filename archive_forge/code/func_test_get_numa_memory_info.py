from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_numa_memory_info(self):
    system_memory = mock.MagicMock()
    system_memory.path_.return_value = 'fake_wmi_obj_path'
    numa_node_memory = mock.MagicMock()
    numa_node_memory.path_.return_value = 'fake_wmi_obj_path1'
    numa_node_assoc = [system_memory]
    memory_info = self._hostutils._get_numa_memory_info(numa_node_assoc, [system_memory, numa_node_memory])
    self.assertEqual(system_memory, memory_info)