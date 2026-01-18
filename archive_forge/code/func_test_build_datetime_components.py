import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_build_datetime_components(self):
    dt = DT(2015, 8, 29, tzinfo=tz.tzutc())
    self.assertEqual(dt, self.eval('datetime(2015, 8, 29)'))
    self.assertEqual(dt, self.eval('datetime(year => 2015, month => 8, day => 29,hour => 0, minute => 0, second => 0, microsecond => 0)'))