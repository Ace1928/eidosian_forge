import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_46_fake_read_lock(self):
    t = self.get_transport()
    lf1 = LockDir(t, 'test_lock')
    lf1.create()
    lf1.lock_read()
    lf1.unlock()