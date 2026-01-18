from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_lv_extend(self):
    self.vg.deactivate_lv = mock.MagicMock()
    self.vg.create_volume('test', '1G')
    self.vg.extend_volume('test', '2G')
    self.vg.deactivate_lv.assert_called_once_with('test')
    self.vg.deactivate_lv.reset_mock()
    self.vg.create_volume('test', '1G')
    self.vg.vg_name = 'test-volumes'
    self.vg.extend_volume('test', '2G')
    self.assertFalse(self.vg.deactivate_lv.called)