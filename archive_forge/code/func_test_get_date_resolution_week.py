import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_get_date_resolution_week(self):
    self.assertEqual(get_date_resolution('2004-W53'), DateResolution.Week)
    self.assertEqual(get_date_resolution('2009-W01'), DateResolution.Week)
    self.assertEqual(get_date_resolution('2004W53'), DateResolution.Week)