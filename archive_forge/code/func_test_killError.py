from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
def test_killError(self) -> None:
    """
        If L{kill} raises an exception other than L{OSError} with errno set to
        C{ESRCH}, the exception is passed up to the caller of
        L{FilesystemLock.lock}.
        """

    def fakeKill(pid: int, signal: int) -> NoReturn:
        raise OSError(errno.EPERM, None)
    self.patch(lockfile, 'kill', fakeKill)
    lockf = self.mktemp()
    lockfile.symlink(str(43125), lockf)
    lock = lockfile.FilesystemLock(lockf)
    exc = self.assertRaises(OSError, lock.lock)
    self.assertEqual(exc.errno, errno.EPERM)
    self.assertFalse(lock.locked)