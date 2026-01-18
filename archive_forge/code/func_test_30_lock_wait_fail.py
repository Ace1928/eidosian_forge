import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_30_lock_wait_fail(self):
    """Wait on a lock, then fail

        We ask to wait up to 400ms; this should fail within at most one
        second.  (Longer times are more realistic but we don't want the test
        suite to take too long, and this should do for now.)
        """
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    lf2 = LockDir(t, 'test_lock')
    self.setup_log_reporter(lf2)
    lf1.attempt_lock()
    try:
        before = time.time()
        self.assertRaises(LockContention, lf2.wait_lock, timeout=0.4, poll=0.1)
        after = time.time()
        self.assertTrue(after - before <= 8.0, 'took %f seconds to detect lock contention' % (after - before))
    finally:
        lf1.unlock()
    self.assertEqual(1, len(self._logged_reports))
    self.assertContainsRe(self._logged_reports[0][0], 'Unable to obtain lock .* held by jrandom@example\\.com on .* \\(process #\\d+\\), acquired .* ago\\.\\nWill continue to try until \\d{2}:\\d{2}:\\d{2}, unless you press Ctrl-C.\\nSee "brz help break-lock" for more.')