import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_build_datetime_iso(self):
    self.assertEqual(DT(2015, 8, 29, tzinfo=tz.tzutc()), self.eval('datetime("2015-8-29")'))
    self.assertEqual(DT(2008, 9, 3, 20, 56, 35, 450686, tzinfo=tz.tzutc()), self.eval('datetime("2008-09-03T20:56:35.450686")'))
    self.assertEqual(DT(2008, 9, 3, 20, 56, 35, 450686, tzinfo=tz.tzutc()), self.eval('datetime("2008-09-03T20:56:35.450686Z")'))
    self.assertEqual(DT(2008, 9, 3, 0, 0, tzinfo=tz.tzutc()), self.eval('datetime("20080903")'))
    dt = self.eval('datetime("2008-09-03T20:56:35.450686+03:00")')
    self.assertEqual(DT(2008, 9, 3, 20, 56, 35, 450686), dt.replace(tzinfo=None))
    self.assertEqual(TS(hours=3), dt.utcoffset())