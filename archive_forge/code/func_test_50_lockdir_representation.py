import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_50_lockdir_representation(self):
    """Check the on-disk representation of LockDirs is as expected.

        There should always be a top-level directory named by the lock.
        When the lock is held, there should be a lockname/held directory
        containing an info file.
        """
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    self.assertTrue(t.has('test_lock'))
    lf1.lock_write()
    self.assertTrue(t.has('test_lock/held/info'))
    lf1.unlock()
    self.assertFalse(t.has('test_lock/held/info'))