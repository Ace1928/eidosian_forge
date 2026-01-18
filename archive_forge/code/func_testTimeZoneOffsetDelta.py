import datetime
import sys
import types
import unittest
import six
from apitools.base.protorpclite import test_util
from apitools.base.protorpclite import util
def testTimeZoneOffsetDelta(self):
    """Test that delta works with TimeZoneOffset."""
    time_zone = util.TimeZoneOffset(datetime.timedelta(minutes=3))
    epoch = time_zone.utcoffset(datetime.datetime.utcfromtimestamp(0))
    self.assertEqual(180, util.total_seconds(epoch))