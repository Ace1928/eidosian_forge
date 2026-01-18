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
@mock.patch('os_brick.utils._time_sleep')
@mock.patch.object(os.path, 'exists')
def test_find_multipath_device_path_mapper(self, exists_mock, sleep_mock):
    exists_mock.side_effect = [False, False, False, True]
    fake_wwn = '1234567890'
    found_path = self.linuxscsi.find_multipath_device_path(fake_wwn)
    expected_path = '/dev/mapper/%s' % fake_wwn
    self.assertEqual(expected_path, found_path)
    self.assertTrue(sleep_mock.called)