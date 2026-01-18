import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_lock_holder_other_user(self):
    """Only auto-break locks held by this user."""
    info = LockHeldInfo.for_this_process(None)
    info.info_dict['user'] = 'notme@example.com'
    info.info_dict['pid'] = '123123123'
    self.assertFalse(info.is_lock_holder_known_dead())