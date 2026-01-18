from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_get_lv_info_notfound(self):
    self.assertEqual([], self.vg.get_lv_info('sudo', vg_name='fake-vg', lv_name='lv-nothere'))
    self.assertEqual([], self.vg.get_lv_info('sudo', vg_name='fake-vg', lv_name='lv-newerror'))