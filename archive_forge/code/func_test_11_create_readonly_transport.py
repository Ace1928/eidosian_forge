import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_11_create_readonly_transport(self):
    """Fail to create lock on readonly transport"""
    t = self.get_readonly_transport()
    lf = LockDir(t, 'test_lock')
    self.assertRaises(LockFailed, lf.create)