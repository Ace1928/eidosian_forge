import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_lock_with_buggy_rename(self):
    t = transport.get_transport_from_url('brokenrename+' + self.get_url())
    ld1 = LockDir(t, 'test_lock')
    ld1.create()
    ld1.attempt_lock()
    ld2 = LockDir(t, 'test_lock')
    e = self.assertRaises(errors.LockContention, ld2.attempt_lock)
    ld1.unlock()
    self.assertEqual([], t.list_dir('test_lock'))