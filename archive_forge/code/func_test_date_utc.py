from datetime import (
import time
import unittest
def test_date_utc(self):
    d = date.today()
    d_utc = datetime(*d.timetuple()[:3]) - self.local_utcoffset
    self.assertEqual(rfc3339(d, utc=True), d_utc.strftime('%Y-%m-%dT%H:%M:%SZ'))