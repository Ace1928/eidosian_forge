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
def test_find_multipath_device_svc(self):

    def fake_execute(*cmd, **kwargs):
        out = "36005076da00638089c000000000004d5 dm-2 IBM,2145\nsize=954M features='1 queue_if_no_path' hwhandler='0' wp=rw\n|-+- policy='round-robin 0' prio=-1 status=active\n| |- 6:0:2:0 sde 8:64  active undef  running\n| `- 6:0:4:0 sdg 8:96  active undef  running\n`-+- policy='round-robin 0' prio=-1 status=enabled\n  |- 6:0:3:0 sdf 8:80  active undef  running\n  `- 6:0:5:0 sdh 8:112 active undef  running\n"
        return (out, None)
    self.linuxscsi._execute = fake_execute
    info = self.linuxscsi.find_multipath_device('/dev/sde')
    self.assertEqual('36005076da00638089c000000000004d5', info['id'])
    self.assertEqual('36005076da00638089c000000000004d5', info['name'])
    self.assertEqual('/dev/mapper/36005076da00638089c000000000004d5', info['device'])
    self.assertEqual('/dev/sde', info['devices'][0]['device'])
    self.assertEqual('6', info['devices'][0]['host'])
    self.assertEqual('0', info['devices'][0]['channel'])
    self.assertEqual('2', info['devices'][0]['id'])
    self.assertEqual('0', info['devices'][0]['lun'])
    self.assertEqual('/dev/sdf', info['devices'][2]['device'])
    self.assertEqual('6', info['devices'][2]['host'])
    self.assertEqual('0', info['devices'][2]['channel'])
    self.assertEqual('3', info['devices'][2]['id'])
    self.assertEqual('0', info['devices'][2]['lun'])