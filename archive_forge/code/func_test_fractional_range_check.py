import datetime
import unittest
from aniso8601 import compat
from aniso8601.builders import (
from aniso8601.builders.python import (
from aniso8601.exceptions import (
from aniso8601.utcoffset import UTCOffset
def test_fractional_range_check(self):
    limit = Limit('Invalid string.', -1, 1, ValueError, 'Value must be between -1..1.', None)
    self.assertEqual(fractional_range_check(10, '1', limit), 1)
    self.assertEqual(fractional_range_check(10, '-1', limit), -1)
    self.assertEqual(fractional_range_check(10, '0.1', limit), FractionalComponent(0, 1))
    self.assertEqual(fractional_range_check(10, '-0.1', limit), FractionalComponent(-0, 1))
    with self.assertRaises(ValueError):
        fractional_range_check(10, '1.1', limit)
    with self.assertRaises(ValueError):
        fractional_range_check(10, '-1.1', limit)