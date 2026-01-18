from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipUnless(platform.isWindows(), 'special rename EIO handling only necessary and correct on Windows.')
def test_lockReleasedDuringAcquireSymlink(self) -> None:
    """
        If the lock is released while an attempt is made to acquire
        it, the lock attempt fails and C{FilesystemLock.lock} returns
        C{False}.  This can happen on Windows when L{lockfile.symlink}
        fails with L{IOError} of C{EIO} because another process is in
        the middle of a call to L{os.rmdir} (implemented in terms of
        RemoveDirectory) which is not atomic.
        """

    def fakeSymlink(src: str, dst: str) -> NoReturn:
        raise OSError(errno.EIO, None)
    self.patch(lockfile, 'symlink', fakeSymlink)
    lockf = self.mktemp()
    lock = lockfile.FilesystemLock(lockf)
    self.assertFalse(lock.lock())
    self.assertFalse(lock.locked)