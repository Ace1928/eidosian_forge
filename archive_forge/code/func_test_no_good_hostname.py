import os
import time
import breezy
from .. import config, errors, lock, lockdir, osutils, tests, transport
from ..errors import (LockBreakMismatch, LockBroken, LockContention,
from ..lockdir import LockDir, LockHeldInfo
from . import TestCase, TestCaseInTempDir, TestCaseWithTransport, features
def test_no_good_hostname(self):
    """Correctly handle ambiguous hostnames.

        If the lock's recorded with just 'localhost' we can't really trust
        it's the same 'localhost'.  (There are quite a few of them. :-)
        So even if the process is known not to be alive, we can't say that's
        known for sure.
        """
    self.overrideAttr(lockdir, 'get_host_name', lambda: 'localhost')
    info = LockHeldInfo.for_this_process(None)
    info.info_dict['pid'] = '123123123'
    self.assertFalse(info.is_lock_holder_known_dead())