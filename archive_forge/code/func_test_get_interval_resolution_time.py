import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_get_interval_resolution_time(self):
    self.assertEqual(get_repeating_interval_resolution('R/P1M/1981-04-05T01'), IntervalResolution.Hours)
    self.assertEqual(get_repeating_interval_resolution('R1/P1M/1981-04-05T01:01'), IntervalResolution.Minutes)
    self.assertEqual(get_repeating_interval_resolution('R2/P1M/1981-04-05T01:01:00'), IntervalResolution.Seconds)
    self.assertEqual(get_repeating_interval_resolution('R/1981-04-05T01/P1M'), IntervalResolution.Hours)
    self.assertEqual(get_repeating_interval_resolution('R1/1981-04-05T01:01/P1M'), IntervalResolution.Minutes)
    self.assertEqual(get_repeating_interval_resolution('R2/1981-04-05T01:01:00/P1M'), IntervalResolution.Seconds)