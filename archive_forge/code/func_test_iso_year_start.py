import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_iso_year_start(self):
    yearstart = PythonTimeBuilder._iso_year_start(2004)
    self.assertEqual(yearstart, datetime.date(year=2003, month=12, day=29))
    yearstart = PythonTimeBuilder._iso_year_start(2010)
    self.assertEqual(yearstart, datetime.date(year=2010, month=1, day=4))
    yearstart = PythonTimeBuilder._iso_year_start(2009)
    self.assertEqual(yearstart, datetime.date(year=2008, month=12, day=29))