from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
@mock.patch('tenacity.nap.sleep', mock.Mock())
@mock.patch.object(brick.putils, 'execute')
def test_get_lv_info_retry(self, exec_mock):
    exec_mock.side_effect = (processutils.ProcessExecutionError('', '', exit_code=139), ('vg name size', ''))
    self.assertEqual([{'name': 'fake-1', 'size': '1.00g', 'vg': 'fake-vg'}, {'name': 'fake-2', 'size': '1.00g', 'vg': 'fake-vg'}], self.vg.get_lv_info('sudo', vg_name='vg', lv_name='name'))