import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_get_interval_resolution_duration(self):
    self.assertEqual(get_repeating_interval_resolution('R/2014-11-12/P1Y2M3D'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R1/2014-11-12/P1Y2M'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R2/2014-11-12/P1Y'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R3/2014-11-12/P1W'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R4/2014-11-12/P1Y2M3DT4H'), IntervalResolution.Hours)
    self.assertEqual(get_repeating_interval_resolution('R5/2014-11-12/P1Y2M3DT4H54M'), IntervalResolution.Minutes)
    self.assertEqual(get_repeating_interval_resolution('R6/2014-11-12/P1Y2M3DT4H54M6S'), IntervalResolution.Seconds)
    self.assertEqual(get_repeating_interval_resolution('R/P1Y2M3D/2014-11-12'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R1/P1Y2M/2014-11-12'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R2/P1Y/2014-11-12'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R3/P1W/2014-11-12'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R4/P1Y2M3DT4H/2014-11-12'), IntervalResolution.Hours)
    self.assertEqual(get_repeating_interval_resolution('R5/P1Y2M3DT4H54M/2014-11-12'), IntervalResolution.Minutes)
    self.assertEqual(get_repeating_interval_resolution('R6/P1Y2M3DT4H54M6S/2014-11-12'), IntervalResolution.Seconds)