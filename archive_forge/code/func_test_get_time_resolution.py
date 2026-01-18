import unittest
import aniso8601
from aniso8601.builders import DatetimeTuple, DateTuple, TimeTuple, TimezoneTuple
from aniso8601.exceptions import ISOFormatError
from aniso8601.resolution import TimeResolution
from aniso8601.tests.compat import mock
from aniso8601.time import (
def test_get_time_resolution(self):
    self.assertEqual(get_time_resolution('01:23:45'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('24:00:00'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('23:21:28,512400'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('23:21:28.512400'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('01:23'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('24:00'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('01:23,4567'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('01:23.4567'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('012345'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('240000'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('0123'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('2400'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('01'), TimeResolution.Hours)
    self.assertEqual(get_time_resolution('24'), TimeResolution.Hours)
    self.assertEqual(get_time_resolution('12,5'), TimeResolution.Hours)
    self.assertEqual(get_time_resolution('12.5'), TimeResolution.Hours)
    self.assertEqual(get_time_resolution('232128.512400+00:00'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('0123.4567+00:00'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('01.4567+00:00'), TimeResolution.Hours)
    self.assertEqual(get_time_resolution('01:23:45+00:00'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('24:00:00+00:00'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('23:21:28.512400+00:00'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('01:23+00:00'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('24:00+00:00'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('01:23.4567+00:00'), TimeResolution.Minutes)
    self.assertEqual(get_time_resolution('23:21:28.512400+11:15'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('23:21:28.512400-12:34'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('23:21:28.512400Z'), TimeResolution.Seconds)
    self.assertEqual(get_time_resolution('06:14:00.000123Z'), TimeResolution.Seconds)