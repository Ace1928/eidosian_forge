from datetime import (
import time
import unittest
def test_microsecond(self):
    x = datetime(2018, 9, 20, 13, 11, 21, 12345)
    self.assertEqual(format_microsecond(datetime(2018, 9, 20, 13, 11, 21, 12345), utc=True, use_system_timezone=False), '2018-09-20T13:11:21.012345Z')