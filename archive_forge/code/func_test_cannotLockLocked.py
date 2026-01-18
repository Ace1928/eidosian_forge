from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_cannotLockLocked(self) -> None:
    """
        If a lock is currently locked, it cannot be locked again.
        """
    lockf = self.mktemp()
    firstLock = lockfile.FilesystemLock(lockf)
    self.assertTrue(firstLock.lock())
    secondLock = lockfile.FilesystemLock(lockf)
    self.assertFalse(secondLock.lock())
    self.assertFalse(secondLock.locked)