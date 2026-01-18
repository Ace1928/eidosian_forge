import builtins
import errno
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
import os_brick.privileged as privsep_brick
import os_brick.privileged.nvmeof as privsep_nvme
from os_brick.privileged import rootwrap
from os_brick.tests import base
@ddt.data(OSError(errno.ENOENT), putils.ProcessExecutionError(exit_code=123))
@mock.patch('os.makedirs')
@mock.patch.object(rootwrap, 'custom_execute')
def test_create_hostnqn_nvme_not_present(self, exception, mock_exec, mock_mkdirs):
    mock_exec.side_effect = exception
    res = privsep_nvme.create_hostnqn()
    mock_mkdirs.assert_called_once_with('/etc/nvme', mode=493, exist_ok=True)
    mock_exec.assert_called_once_with('nvme', 'show-hostnqn')
    self.assertEqual('', res)