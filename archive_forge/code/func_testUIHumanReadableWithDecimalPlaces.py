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
def testUIHumanReadableWithDecimalPlaces(self):
    """Tests HumanReadableWithDecimalPlaces for UI."""
    self.assertEqual('1.0 GiB', HumanReadableWithDecimalPlaces(1024 ** 3 + 1024 ** 2 * 10, 1))
    self.assertEqual('1.0 GiB', HumanReadableWithDecimalPlaces(1024 ** 3), 1)
    self.assertEqual('1.01 GiB', HumanReadableWithDecimalPlaces(1024 ** 3 + 1024 ** 2 * 10, 2))
    self.assertEqual('1.000 GiB', HumanReadableWithDecimalPlaces(1024 ** 3 + 1024 ** 2 * 5, 3))
    self.assertEqual('1.10 GiB', HumanReadableWithDecimalPlaces(1024 ** 3 + 1024 ** 2 * 100, 2))
    self.assertEqual('1.100 GiB', HumanReadableWithDecimalPlaces(1024 ** 3 + 1024 ** 2 * 100, 3))
    self.assertEqual('10.00 MiB', HumanReadableWithDecimalPlaces(1024 ** 2 * 10, 2))
    self.assertEqual('2.01 GiB', HumanReadableWithDecimalPlaces(2157969408, 2))
    self.assertEqual('2.0 GiB', HumanReadableWithDecimalPlaces(2157969408, 1))
    self.assertEqual('0 B', HumanReadableWithDecimalPlaces(0, 0))
    self.assertEqual('0.00 B', HumanReadableWithDecimalPlaces(0, 2))
    self.assertEqual('0.00000 B', HumanReadableWithDecimalPlaces(0, 5))