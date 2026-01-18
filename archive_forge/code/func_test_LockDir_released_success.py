import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_LockDir_released_success(self):
    LockDir.hooks.install_named_hook('lock_released', self.record_hook, 'record_hook')
    ld = self.get_lock()
    ld.create()
    self.assertEqual([], self._calls)
    result = ld.attempt_lock()
    self.assertEqual([], self._calls)
    ld.unlock()
    lock_path = ld.transport.abspath(ld.path)
    self.assertEqual([lock.LockResult(lock_path, result)], self._calls)