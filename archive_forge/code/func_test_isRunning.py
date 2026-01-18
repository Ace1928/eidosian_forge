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
def test_isRunning(self) -> None:
    """
        L{NonePIDFile.isRunning} returns L{False}.
        """
    pidFile = NonePIDFile()
    self.assertEqual(pidFile.isRunning(), False)