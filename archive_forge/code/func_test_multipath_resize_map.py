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
def test_multipath_resize_map(self):
    dm_path = '/dev/dm-5'
    self.linuxscsi.multipath_resize_map(dm_path)
    expected_commands = ['multipathd resize map %s' % dm_path]
    self.assertEqual(expected_commands, self.cmds)