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
def test_formatWithPID(self) -> None:
    """
        L{PIDFile._format} returns the expected format when given a PID.
        """
    self.assertEqual(PIDFile._format(pid=1337), b'1337\n')