from datetime import (
import time
import unittest
def testPDTChange(self):
    """Test Daylight saving change"""
    self.assertEqual(rfc3339(datetime(2010, 3, 14, 1, 59)), '2010-03-14T01:59:00-08:00')
    self.assertEqual(rfc3339(datetime(2010, 3, 14, 3, 0)), '2010-03-14T03:00:00-07:00')