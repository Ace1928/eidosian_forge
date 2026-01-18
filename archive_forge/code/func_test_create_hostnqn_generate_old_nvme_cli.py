import builtins
import errno
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
import os_brick.privileged as privsep_brick
import os_brick.privileged.nvmeof as privsep_nvme
from os_brick.privileged import rootwrap
from os_brick.tests import base
@ddt.data((231, 'error: Invalid sub-command\n', ''), (254, '', 'hostnqn is not available -- use nvme gen-hostnqn\n'))
@ddt.unpack
@mock.patch('os.chmod')
@mock.patch.object(builtins, 'open', new_callable=mock.mock_open)
@mock.patch('os.makedirs')
@mock.patch.object(rootwrap, 'custom_execute')
def test_create_hostnqn_generate_old_nvme_cli(self, exit_code, stdout, stderr, mock_exec, mock_mkdirs, mock_open, mock_chmod):
    hostnqn = mock.Mock()
    mock_exec.side_effect = [putils.ProcessExecutionError(exit_code=exit_code, stdout=stdout, stderr=stderr), (hostnqn, mock.sentinel.err)]
    res = privsep_nvme.create_hostnqn()
    mock_mkdirs.assert_called_once_with('/etc/nvme', mode=493, exist_ok=True)
    self.assertEqual(2, mock_exec.call_count)
    mock_exec.assert_has_calls([mock.call('nvme', 'show-hostnqn'), mock.call('nvme', 'gen-hostnqn')])
    mock_open.assert_called_once_with('/etc/nvme/hostnqn', 'w')
    stripped_hostnqn = hostnqn.strip.return_value
    mock_open().write.assert_called_once_with(stripped_hostnqn)
    mock_chmod.assert_called_once_with('/etc/nvme/hostnqn', 420)
    self.assertEqual(stripped_hostnqn, res)