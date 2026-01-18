import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_corrupt_lockdir_info(self):
    """We can cope with corrupt (and thus unparseable) info files."""
    t = self.get_transport()
    t.mkdir('test_lock')
    t.mkdir('test_lock/held')
    t.put_bytes('test_lock/held/info', b'\x00')
    lf = LockDir(t, 'test_lock')
    self.assertRaises(errors.LockCorrupt, lf.peek)
    self.assertRaises((errors.LockCorrupt, errors.LockContention), lf.attempt_lock)
    self.assertRaises(errors.LockCorrupt, lf.validate_token, 'fake token')