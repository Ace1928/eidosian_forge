import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_get_time_resolution_internal(self):
    self.assertEqual(_get_time_resolution(TimeTuple(hh='01', mm='02', ss='03', tz=None)), TimeResolution.Seconds)
    self.assertEqual(_get_time_resolution(TimeTuple(hh='01', mm='02', ss=None, tz=None)), TimeResolution.Minutes)
    self.assertEqual(_get_time_resolution(TimeTuple(hh='01', mm=None, ss=None, tz=None)), TimeResolution.Hours)