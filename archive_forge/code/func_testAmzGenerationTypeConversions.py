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
def testAmzGenerationTypeConversions(self):
    amz_gen_as_str = six.ensure_str('9PpsRjBGjBh90IvIS96dgRc_UL6NyGqD')
    amz_gen_as_long = 25923956239092482442895228561437790190304192615858167521375267910356975448388
    self.assertEqual(text_util.DecodeLongAsString(amz_gen_as_long), amz_gen_as_str)
    self.assertEqual(text_util.EncodeStringAsLong(amz_gen_as_str), amz_gen_as_long)