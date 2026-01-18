import binascii
import copy
from unittest import mock
from castellan.common.objects import symmetric_key as key
from castellan.tests.unit.key_manager import fake
from oslo_concurrency import processutils as putils
from os_brick.encryptors import luks
from os_brick import exception
from os_brick.tests.encryptors import test_base
@mock.patch('os_brick.executor.Executor._execute')
@mock.patch('os_brick.encryptors.luks.LOG')
def test_is_luks_with_error(self, mock_log, mock_execute):
    error_msg = 'Device %s is not a valid LUKS device.' % self.dev_path
    mock_execute.side_effect = putils.ProcessExecutionError(exit_code=1, stderr=error_msg)
    luks.is_luks(self.root_helper, self.dev_path, execute=mock_execute)
    mock_execute.assert_has_calls([mock.call('cryptsetup', 'isLuks', '--verbose', self.dev_path, run_as_root=True, root_helper=self.root_helper, check_exit_code=True)])
    self.assertEqual(1, mock_log.warning.call_count)