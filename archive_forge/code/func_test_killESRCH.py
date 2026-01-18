from __future__ import annotations
import errno
import os
from unittest import skipIf, skipUnless
from typing_extensions import NoReturn
from twisted.python import lockfile
from twisted.python.reflect import requireModule
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
@skipIf(skipKill, skipKillReason)
def test_killESRCH(self) -> None:
    """
        L{lockfile.kill} raises L{OSError} with errno of L{ESRCH} if
        passed a PID which does not correspond to any process.
        """
    exc = self.assertRaises(OSError, lockfile.kill, 2 ** 31 - 1, 0)
    self.assertEqual(exc.errno, errno.ESRCH)