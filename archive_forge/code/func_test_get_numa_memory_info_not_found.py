from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_numa_memory_info_not_found(self):
    other = mock.MagicMock()
    memory_info = self._hostutils._get_numa_memory_info([], [other])
    self.assertIsNone(memory_info)