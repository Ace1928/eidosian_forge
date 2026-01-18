import multiprocessing
from unittest import mock
from neutron_lib.tests import _base as base
from neutron_lib.utils import host
@mock.patch.object(multiprocessing, 'cpu_count', side_effect=NotImplementedError())
def test_cpu_count_not_implemented(self, mock_cpu_count):
    self.assertEqual(1, host.cpu_count())
    mock_cpu_count.assert_called_once_with()