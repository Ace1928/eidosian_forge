from unittest import mock
import ddt
from oslo_concurrency import processutils
from os_brick import exception
from os_brick.initiator.windows import rbd
from os_brick.tests.initiator.connectors import test_base_rbd
from os_brick.tests.windows import test_base
@ddt.data(True, False)
def test_check_rbd(self, rbd_available):
    self._execute.side_effect = None if rbd_available else processutils.ProcessExecutionError
    self.assertEqual(rbd_available, self._conn._check_rbd())
    if rbd_available:
        self._conn._ensure_rbd_available()
    else:
        self.assertRaises(exception.BrickException, self._conn._ensure_rbd_available)
    expected_cmd = ['where.exe', 'rbd']
    self._execute.assert_any_call(*expected_cmd)