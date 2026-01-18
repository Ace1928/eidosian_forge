import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_datetime_fields(self):
    dt = DT(2006, 11, 21, 16, 30, tzinfo=tz.tzutc())
    self.assertEqual(2006, self.eval('$.year', dt))
    self.assertEqual(11, self.eval('$.month', dt))
    self.assertEqual(21, self.eval('$.day', dt))
    self.assertEqual(16, self.eval('$.hour', dt))
    self.assertEqual(30, self.eval('$.minute', dt))
    self.assertEqual(0, self.eval('$.second', dt))
    self.assertEqual(0, self.eval('$.microsecond', dt))
    self.assertEqual(1164126600, self.eval('$.timestamp', dt))
    self.assertEqual(1, self.eval('$.weekday', dt))
    self.assertEqual(TS(), self.eval('$.offset', dt))
    self.assertEqual(TS(hours=16, minutes=30), self.eval('$.time', dt))
    self.assertEqual(dt.replace(hour=0, minute=0), self.eval('$.date', dt))
    self.assertEqual(dt, self.eval('$.utc', dt))