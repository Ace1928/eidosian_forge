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
def testUIPrettyTime(self):
    """Tests PrettyTime for UI."""
    self.assertEqual('25:02:10', PrettyTime(90130))
    self.assertEqual('01:00:00', PrettyTime(3600))
    self.assertEqual('00:59:59', PrettyTime(3599))
    self.assertEqual('100+ hrs', PrettyTime(3600 * 100))
    self.assertEqual('999+ hrs', PrettyTime(3600 * 10000))