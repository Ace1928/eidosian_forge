import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_03_readonly_peek(self):
    lf = LockDir(self.get_readonly_transport(), 'test_lock')
    self.assertEqual(lf.peek(), None)