import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_auto_break_stale_lock(self):
    """Locks safely known to be stale are just cleaned up.

        This generates a warning but no other user interaction.
        """
    self.overrideAttr(lockdir, 'get_host_name', lambda: 'aproperhostname')
    l1 = LockDir(self.get_transport(), 'a', extra_holder_info={'pid': '12312313'})
    token_1 = l1.attempt_lock()
    l2 = LockDir(self.get_transport(), 'a')
    token_2 = l2.attempt_lock()
    self.assertRaises(errors.LockBroken, l1.unlock)
    l2.unlock()