from unittest import mock
from oslo_concurrency import processutils
from os_brick import exception
from os_brick import executor as os_brick_executor
from os_brick.local_dev import lvm as brick
from os_brick.privileged import rootwrap as priv_rootwrap
from os_brick.tests import base
def test_get_all_physical_volumes(self):
    pvs = self.vg.get_all_physical_volumes('sudo', 'fake-vg')
    self.assertEqual(3, len(pvs))
    pvs = self.vg.get_all_physical_volumes('sudo')
    self.assertEqual(4, len(pvs))