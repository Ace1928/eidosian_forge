from datetime import (
import time
import unittest
def test_1920(self):
    d = date(1920, 2, 29)
    x = rfc3339(d, utc=False, use_system_timezone=True)
    self.assertTrue(x.startswith('1920-02-29T00:00:00'))