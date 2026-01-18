from unittest import mock
from os_brick.initiator import utils
from os_brick.tests import base
@mock.patch('os.name', 'posix')
@mock.patch('oslo_concurrency.processutils.execute', side_effect=utils.putils.ProcessExecutionError)
def test_check_manual_scan_not_supported(self, mock_exec):
    self.assertFalse(utils.check_manual_scan())
    mock_exec.assert_called_once_with('grep', '-F', 'node.session.scan', '/sbin/iscsiadm')