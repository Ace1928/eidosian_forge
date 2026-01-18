import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_display_form(self):
    ld1 = self.get_lock()
    ld1.create()
    ld1.lock_write()
    try:
        info_list = ld1.peek().to_readable_dict()
    finally:
        ld1.unlock()
    self.assertEqual(info_list['user'], 'jrandom@example.com')
    self.assertIsInstance(info_list['pid'], int)
    self.assertContainsRe(info_list['time_ago'], '^\\d+ seconds? ago$')