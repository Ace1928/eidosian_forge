from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_isLocked(self) -> None:
    """
        L{isLocked} returns C{True} if the named lock is currently locked,
        C{False} otherwise.
        """
    lockf = self.mktemp()
    self.assertFalse(lockfile.isLocked(lockf))
    lock = lockfile.FilesystemLock(lockf)
    self.assertTrue(lock.lock())
    self.assertTrue(lockfile.isLocked(lockf))
    lock.unlock()
    self.assertFalse(lockfile.isLocked(lockf))