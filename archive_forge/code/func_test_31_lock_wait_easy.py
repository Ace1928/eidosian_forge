import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_31_lock_wait_easy(self):
    """Succeed when waiting on a lock with no contention.
        """
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    self.setup_log_reporter(lf1)
    try:
        before = time.time()
        lf1.wait_lock(timeout=0.4, poll=0.1)
        after = time.time()
        self.assertTrue(after - before <= 1.0)
    finally:
        lf1.unlock()
    self.assertEqual([], self._logged_reports)