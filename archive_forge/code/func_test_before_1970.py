from datetime import (
import time
import unittest
def test_before_1970(self):
    d = date(1885, 1, 4)
    self.assertTrue(rfc3339(d).startswith('1885-01-04T00:00:00'))
    self.assertEqual(rfc3339(d, utc=True, use_system_timezone=False), '1885-01-04T00:00:00Z')