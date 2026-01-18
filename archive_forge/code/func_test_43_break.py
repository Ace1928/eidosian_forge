import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_43_break(self):
    """Break a lock whose caller has forgotten it"""
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    lf1.attempt_lock()
    del lf1
    lf2 = LockDir(t, 'test_lock')
    holder_info = lf2.peek()
    self.assertTrue(holder_info)
    lf2.force_break(holder_info)
    lf2.attempt_lock()
    self.addCleanup(lf2.unlock)
    lf2.confirm()