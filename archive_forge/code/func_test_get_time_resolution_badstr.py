import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_get_time_resolution_badstr(self):
    testtuples = ('A6:14:00.000123Z', '06:14:0B', 'bad', '')
    for testtuple in testtuples:
        with self.assertRaises(ISOFormatError):
            get_time_resolution(testtuple)