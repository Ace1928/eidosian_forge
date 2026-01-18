import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_lock_permission(self):
    self.requireFeature(features.not_running_as_root)
    if not osutils.supports_posix_readonly():
        raise tests.TestSkipped('Cannot induce a permission failure')
    ld1 = self.get_lock()
    lock_path = ld1.transport.local_abspath('test_lock')
    os.mkdir(lock_path)
    osutils.make_readonly(lock_path)
    self.assertRaises(errors.LockFailed, ld1.attempt_lock)