import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_build_week_date(self):
    weekdate = PythonTimeBuilder._build_week_date(2009, 1)
    self.assertEqual(weekdate, datetime.date(year=2008, month=12, day=29))
    weekdate = PythonTimeBuilder._build_week_date(2009, 53, isoday=7)
    self.assertEqual(weekdate, datetime.date(year=2010, month=1, day=3))