import errno
from functools import wraps
from os import getpid, name as SYSTEM_NAME
from typing import Any, Callable, Optional
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
import twisted.trial.unittest
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest
from ...runner import _pidfile
from .._pidfile import (
@ifPlatformSupported
def test_contextManagerDoesntExist(self) -> None:
    """
        When used as a context manager, a L{PIDFile} will replace the
        underlying PIDFile rather than raising L{AlreadyRunningError} if the
        contained PID file exists but refers to a non-running PID.
        """
    pidFile = PIDFile(self.filePath())
    pidFile._write(1337)

    def kill(pid: int, signal: int) -> None:
        raise OSError(errno.ESRCH, 'No such process')
    self.patch(_pidfile, 'kill', kill)
    e = self.assertRaises(StalePIDFileError, pidFile.isRunning)
    self.assertEqual(str(e), 'PID file refers to non-existing process')
    with pidFile:
        self.assertEqual(pidFile.read(), getpid())