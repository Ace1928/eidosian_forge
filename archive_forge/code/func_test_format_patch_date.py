import random
import time
from breezy import tests, timestamp
from breezy.osutils import local_time_offset
def test_format_patch_date(self):
    self.assertEqual('1970-01-01 00:00:00 +0000', timestamp.format_patch_date(0))
    self.assertEqual('1970-01-01 00:00:00 +0000', timestamp.format_patch_date(0, 5 * 3600))
    self.assertEqual('1970-01-01 00:00:00 +0000', timestamp.format_patch_date(0, -5 * 3600))
    self.assertEqual('2007-03-06 10:04:19 -0500', timestamp.format_patch_date(1173193459, -5 * 3600))
    self.assertEqual('2007-03-06 09:34:19 -0530', timestamp.format_patch_date(1173193459, -5.5 * 3600))
    self.assertEqual('2007-03-06 15:05:19 +0001', timestamp.format_patch_date(1173193459, +1 * 60))