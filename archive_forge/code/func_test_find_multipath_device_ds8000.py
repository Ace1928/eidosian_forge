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
def test_find_multipath_device_ds8000(self):

    def fake_execute(*cmd, **kwargs):
        out = "36005076303ffc48e0000000000000101 dm-2 IBM,2107900\nsize=1.0G features='1 queue_if_no_path' hwhandler='0' wp=rw\n`-+- policy='round-robin 0' prio=-1 status=active\n  |- 6:0:2:0  sdd 8:64  active undef  running\n  `- 6:1:0:3  sdc 8:32  active undef  running\n"
        return (out, None)
    self.linuxscsi._execute = fake_execute
    info = self.linuxscsi.find_multipath_device('/dev/sdd')
    self.assertEqual('36005076303ffc48e0000000000000101', info['id'])
    self.assertEqual('36005076303ffc48e0000000000000101', info['name'])
    self.assertEqual('/dev/mapper/36005076303ffc48e0000000000000101', info['device'])
    self.assertEqual('/dev/sdd', info['devices'][0]['device'])
    self.assertEqual('6', info['devices'][0]['host'])
    self.assertEqual('0', info['devices'][0]['channel'])
    self.assertEqual('2', info['devices'][0]['id'])
    self.assertEqual('0', info['devices'][0]['lun'])
    self.assertEqual('/dev/sdc', info['devices'][1]['device'])
    self.assertEqual('6', info['devices'][1]['host'])
    self.assertEqual('1', info['devices'][1]['channel'])
    self.assertEqual('0', info['devices'][1]['id'])
    self.assertEqual('3', info['devices'][1]['lun'])