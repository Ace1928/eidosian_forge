import os
import os.path
import textwrap
from unittest import mock
import ddt
from oslo_concurrency import processutils as putils
from oslo_log import log as logging
from os_brick import exception
from os_brick.initiator import linuxscsi
from os_brick.tests import base
@mock.patch.object(os.path, 'exists', return_value=True)
def test_find_multipath_device_path(self, exists_mock):
    fake_wwn = '1234567890'
    found_path = self.linuxscsi.find_multipath_device_path(fake_wwn)
    expected_path = '/dev/disk/by-id/dm-uuid-mpath-%s' % fake_wwn
    self.assertEqual(expected_path, found_path)