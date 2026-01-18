import unittest
import aniso8601
from aniso8601.builders import (
from aniso8601.exceptions import ISOFormatError
from aniso8601.interval import (
from aniso8601.resolution import IntervalResolution
from aniso8601.tests.compat import mock
def test_get_interval_resolution_date(self):
    self.assertEqual(get_repeating_interval_resolution('R/P1.5Y/2018'), IntervalResolution.Year)
    self.assertEqual(get_repeating_interval_resolution('R1/P1.5Y/2018-03'), IntervalResolution.Month)
    self.assertEqual(get_repeating_interval_resolution('R2/P1.5Y/2018-03-06'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R3/P1.5Y/2018W01'), IntervalResolution.Week)
    self.assertEqual(get_repeating_interval_resolution('R4/P1.5Y/2018-306'), IntervalResolution.Ordinal)
    self.assertEqual(get_repeating_interval_resolution('R5/P1.5Y/2018W012'), IntervalResolution.Weekday)
    self.assertEqual(get_repeating_interval_resolution('R/2018/P1.5Y'), IntervalResolution.Year)
    self.assertEqual(get_repeating_interval_resolution('R1/2018-03/P1.5Y'), IntervalResolution.Month)
    self.assertEqual(get_repeating_interval_resolution('R2/2018-03-06/P1.5Y'), IntervalResolution.Day)
    self.assertEqual(get_repeating_interval_resolution('R3/2018W01/P1.5Y'), IntervalResolution.Week)
    self.assertEqual(get_repeating_interval_resolution('R4/2018-306/P1.5Y'), IntervalResolution.Ordinal)
    self.assertEqual(get_repeating_interval_resolution('R5/2018W012/P1.5Y'), IntervalResolution.Weekday)