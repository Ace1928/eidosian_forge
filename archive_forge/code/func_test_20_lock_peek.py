import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_20_lock_peek(self):
    """Peek at the state of a lock"""
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    lf1.attempt_lock()
    self.addCleanup(lf1.unlock)
    info1 = lf1.peek()
    self.assertEqual(set(info1.info_dict.keys()), {'user', 'nonce', 'hostname', 'pid', 'start_time'})
    info2 = LockDir(t, 'test_lock').peek()
    self.assertEqual(info1, info2)
    self.assertEqual(LockDir(t, 'other_lock').peek(), None)