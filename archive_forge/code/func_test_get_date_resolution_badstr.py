import unittest
import aniso8601
from aniso8601.date import get_date_resolution, parse_date
from aniso8601.exceptions import DayOutOfBoundsError, ISOFormatError
from aniso8601.resolution import DateResolution
from aniso8601.tests.compat import mock
def test_get_date_resolution_badstr(self):
    testtuples = ('W53', '2004-W', '2014-01-230', '2014-012-23', '201-01-23', '201401230', '201401', '')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            get_date_resolution(testtuple)