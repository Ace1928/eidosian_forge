import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_20_lock_contested(self):
    """Contention to get a lock"""
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    lf1.attempt_lock()
    lf2 = LockDir(t, 'test_lock')
    try:
        lf2.attempt_lock()
        self.fail('Failed to detect lock collision')
    except LockContention as e:
        self.assertEqual(e.lock, lf2)
        self.assertContainsRe(str(e), '^Could not acquire.*test_lock.*$')
    lf1.unlock()