from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_activate_lv(self):
    self.vg._supports_lvchange_ignoreskipactivation = True
    with mock.patch.object(self.vg, '_execute') as mock_exec:
        self.vg.activate_lv('my-lv')
        expected = [mock.call('lvchange', '-a', 'y', '--yes', '-K', 'fake-vg/my-lv', root_helper='sudo', run_as_root=True)]
        self.assertEqual(expected, mock_exec.call_args_list)