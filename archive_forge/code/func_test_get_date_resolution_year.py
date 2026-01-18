import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_get_date_resolution_year(self):
    self.assertEqual(get_date_resolution('2013'), DateResolution.Year)
    self.assertEqual(get_date_resolution('0001'), DateResolution.Year)
    self.assertEqual(get_date_resolution('19'), DateResolution.Year)