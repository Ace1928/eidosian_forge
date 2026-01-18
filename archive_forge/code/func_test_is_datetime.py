import datetime
import time
from dateutil import tz
from testtools import matchers
import yaql.tests
def test_is_datetime(self):
    self.assertTrue(self.eval('isDatetime(datetime("2015-8-29"))'))
    self.assertFalse(self.eval('isDatetime(123)'))
    self.assertFalse(self.eval('isDatetime(abc)'))