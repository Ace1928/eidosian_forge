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
def testHumanReadableToBytes(self):
    """Tests converting human-readable strings to byte counts."""
    self.assertEqual(unit_util.HumanReadableToBytes('1'), 1)
    self.assertEqual(unit_util.HumanReadableToBytes('15'), 15)
    self.assertEqual(unit_util.HumanReadableToBytes('15.3'), 15)
    self.assertEqual(unit_util.HumanReadableToBytes('15.7'), 16)
    self.assertEqual(unit_util.HumanReadableToBytes('1023'), 1023)
    self.assertEqual(unit_util.HumanReadableToBytes('1k'), 1024)
    self.assertEqual(unit_util.HumanReadableToBytes('2048'), 2048)
    self.assertEqual(unit_util.HumanReadableToBytes('1 k'), 1024)
    self.assertEqual(unit_util.HumanReadableToBytes('1 K'), 1024)
    self.assertEqual(unit_util.HumanReadableToBytes('1 KB'), 1024)
    self.assertEqual(unit_util.HumanReadableToBytes('1 KiB'), 1024)
    self.assertEqual(unit_util.HumanReadableToBytes('1 m'), 1024 ** 2)
    self.assertEqual(unit_util.HumanReadableToBytes('1 M'), 1024 ** 2)
    self.assertEqual(unit_util.HumanReadableToBytes('1 MB'), 1024 ** 2)
    self.assertEqual(unit_util.HumanReadableToBytes('1 MiB'), 1024 ** 2)
    self.assertEqual(unit_util.HumanReadableToBytes('1 g'), 1024 ** 3)
    self.assertEqual(unit_util.HumanReadableToBytes('1 G'), 1024 ** 3)
    self.assertEqual(unit_util.HumanReadableToBytes('1 GB'), 1024 ** 3)
    self.assertEqual(unit_util.HumanReadableToBytes('1 GiB'), 1024 ** 3)
    self.assertEqual(unit_util.HumanReadableToBytes('1t'), 1024 ** 4)
    self.assertEqual(unit_util.HumanReadableToBytes('1T'), 1024 ** 4)
    self.assertEqual(unit_util.HumanReadableToBytes('1TB'), 1024 ** 4)
    self.assertEqual(unit_util.HumanReadableToBytes('1TiB'), 1024 ** 4)
    self.assertEqual(unit_util.HumanReadableToBytes('1\t   p'), 1024 ** 5)
    self.assertEqual(unit_util.HumanReadableToBytes('1\t   P'), 1024 ** 5)
    self.assertEqual(unit_util.HumanReadableToBytes('1\t   PB'), 1024 ** 5)
    self.assertEqual(unit_util.HumanReadableToBytes('1\t   PiB'), 1024 ** 5)
    self.assertEqual(unit_util.HumanReadableToBytes('1e'), 1024 ** 6)
    self.assertEqual(unit_util.HumanReadableToBytes('1E'), 1024 ** 6)
    self.assertEqual(unit_util.HumanReadableToBytes('1EB'), 1024 ** 6)
    self.assertEqual(unit_util.HumanReadableToBytes('1EiB'), 1024 ** 6)