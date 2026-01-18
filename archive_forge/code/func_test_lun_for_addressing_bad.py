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
@ddt.data(1, 'SAM3', 'TRANSPARENT', 'sam', 'sam2')
def test_lun_for_addressing_bad(self, mode):
    self.assertRaises(exception.InvalidParameterValue, self.linuxscsi.lun_for_addressing, 1, mode)