import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_get_date_resolution_day(self):
    self.assertEqual(get_date_resolution('2004-04-11'), DateResolution.Day)
    self.assertEqual(get_date_resolution('20090121'), DateResolution.Day)