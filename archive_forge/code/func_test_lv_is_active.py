from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_lv_is_active(self):
    self.vg.create_volume('test', '1G')
    with mock.patch.object(self.vg, '_execute', return_value=['owi-a---', '']):
        self.assertTrue(self.vg._lv_is_active('test'))
    with mock.patch.object(self.vg, '_execute', return_value=['owi-----', '']):
        self.assertFalse(self.vg._lv_is_active('test'))