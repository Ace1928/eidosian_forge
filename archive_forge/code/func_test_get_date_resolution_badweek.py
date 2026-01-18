import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_get_date_resolution_badweek(self):
    testtuples = ('2004-W1', '2004W1')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            get_date_resolution(testtuple)