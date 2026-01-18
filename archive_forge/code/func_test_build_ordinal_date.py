import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_build_ordinal_date(self):
    ordinaldate = PythonTimeBuilder._build_ordinal_date(1981, 95)
    self.assertEqual(ordinaldate, datetime.date(year=1981, month=4, day=5))