import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_missing_lockdir_info(self):
    """We can cope with absent info files."""
    t = self.get_transport()
    t.mkdir('test_lock')
    t.mkdir('test_lock/held')
    lf = LockDir(t, 'test_lock')
    self.assertEqual(None, lf.peek())
    try:
        lf.attempt_lock()
    except LockContention:
        pass
    else:
        lf.unlock()
    self.assertRaises((errors.TokenMismatch, errors.LockCorrupt), lf.validate_token, 'fake token')