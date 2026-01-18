import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testDecodeDateTimeWithTruncateTime(self):
    """Test that nanosec time is truncated with truncate_time flag."""
    decoded = util.decode_datetime('2012-09-30T15:31:50.262343123', truncate_time=True)
    expected = datetime.datetime(2012, 9, 30, 15, 31, 50, 262343)
    self.assertEquals(expected, decoded)