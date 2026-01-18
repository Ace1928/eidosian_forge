from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch('tenacity.nap.sleep', mock.Mock())
@mock.patch.object(os_brick_executor.Executor, '_execute')
def test_get_thin_pool_free_space_retry(self, exec_mock):
    exec_mock.side_effect = (processutils.ProcessExecutionError('', '', exit_code=139), ('15.84:50', ''))
    self.assertEqual(7.92, self.vg._get_thin_pool_free_space('vg', 'thinpool'))
    self.assertEqual(2, exec_mock.call_count)
    args = ['env', 'LC_ALL=C', 'lvs', '--noheadings', '--unit=g', '-o', 'size,data_percent', '--separator', ':', '--nosuffix', '/dev/vg/thinpool']
    if self.configuration.lvm_suppress_fd_warnings:
        args.insert(2, 'LVM_SUPPRESS_FD_WARNINGS=1')
    lvs_call = mock.call(*args, root_helper='sudo', run_as_root=True)
    exec_mock.assert_has_calls([lvs_call, lvs_call])