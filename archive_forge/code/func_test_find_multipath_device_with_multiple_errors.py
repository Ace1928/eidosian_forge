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
def test_find_multipath_device_with_multiple_errors(self):

    def fake_execute(*cmd, **kwargs):
        out = "Jun 21 04:39:26 | 8:160: path wwid appears to have changed. Using old wwid.\n\nJun 21 04:39:26 | 65:208: path wwid appears to have changed. Using old wwid.\n\nJun 21 04:39:26 | 65:208: path wwid appears to have changed. Using old wwid.\n3624a93707edcfde1127040370004ee62 dm-84 PURE    ,FlashArray\nsize=100G features='0' hwhandler='0' wp=rw\n`-+- policy='queue-length 0' prio=1 status=active\n  |- 8:0:0:9  sdaa 65:160 active ready running\n  `- 8:0:1:9  sdac 65:192 active ready running\n"
        return (out, None)
    self.linuxscsi._execute = fake_execute
    info = self.linuxscsi.find_multipath_device('/dev/sdaa')
    self.assertEqual('3624a93707edcfde1127040370004ee62', info['id'])
    self.assertEqual('3624a93707edcfde1127040370004ee62', info['name'])
    self.assertEqual('/dev/mapper/3624a93707edcfde1127040370004ee62', info['device'])
    self.assertEqual('/dev/sdaa', info['devices'][0]['device'])
    self.assertEqual('8', info['devices'][0]['host'])
    self.assertEqual('0', info['devices'][0]['channel'])
    self.assertEqual('0', info['devices'][0]['id'])
    self.assertEqual('9', info['devices'][0]['lun'])
    self.assertEqual('/dev/sdac', info['devices'][1]['device'])
    self.assertEqual('8', info['devices'][1]['host'])
    self.assertEqual('0', info['devices'][1]['channel'])
    self.assertEqual('1', info['devices'][1]['id'])
    self.assertEqual('9', info['devices'][1]['lun'])