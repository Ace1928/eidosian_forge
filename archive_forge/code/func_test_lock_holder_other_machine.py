import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_lock_holder_other_machine(self):
    """The lock holder isn't here so we don't know if they're alive."""
    info = LockHeldInfo.for_this_process(None)
    info.info_dict['hostname'] = 'egg.example.com'
    info.info_dict['pid'] = '123123123'
    self.assertFalse(info.is_lock_holder_known_dead())