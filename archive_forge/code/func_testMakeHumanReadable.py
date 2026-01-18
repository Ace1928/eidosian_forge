from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from gslib.utils import boto_util
from gslib.utils import constants
from gslib.utils import system_util
from gslib.utils import ls_helper
from gslib.utils import retry_util
from gslib.utils import text_util
from gslib.utils import unit_util
import gslib.tests.testcase as testcase
from gslib.tests.util import SetEnvironmentForTest
from gslib.tests.util import TestParams
from gslib.utils.text_util import CompareVersions
from gslib.utils.unit_util import DecimalShort
from gslib.utils.unit_util import HumanReadableWithDecimalPlaces
from gslib.utils.unit_util import PrettyTime
import httplib2
import os
import six
from six import add_move, MovedModule
from six.moves import mock
def testMakeHumanReadable(self):
    """Tests converting byte counts to human-readable strings."""
    self.assertEqual(unit_util.MakeHumanReadable(0), '0 B')
    self.assertEqual(unit_util.MakeHumanReadable(1023), '1023 B')
    self.assertEqual(unit_util.MakeHumanReadable(1024), '1 KiB')
    self.assertEqual(unit_util.MakeHumanReadable(1024 ** 2), '1 MiB')
    self.assertEqual(unit_util.MakeHumanReadable(1024 ** 3), '1 GiB')
    self.assertEqual(unit_util.MakeHumanReadable(1024 ** 3 * 5.3), '5.3 GiB')
    self.assertEqual(unit_util.MakeHumanReadable(1024 ** 4 * 2.7), '2.7 TiB')
    self.assertEqual(unit_util.MakeHumanReadable(1024 ** 5), '1 PiB')
    self.assertEqual(unit_util.MakeHumanReadable(1024 ** 6), '1 EiB')