from unittest import mock
from os_win import constants
from os_win import exceptions
from os_win.tests.unit import test_base
from os_win.utils import hostutils
def test_get_cpus_info(self):
    cpu = mock.MagicMock(spec=FakeCPUSpec)
    self._hostutils._conn_cimv2.query.return_value = [cpu]
    cpu_list = self._hostutils.get_cpus_info()
    self.assertEqual([cpu._mock_children], cpu_list)