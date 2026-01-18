from __future__ import annotations
import errno
import io
import os
import pickle
import stat
import sys
import time
from pprint import pformat
from typing import IO, AnyStr, Callable, Dict, List, Optional, Tuple, TypeVar, Union
from unittest import skipIf
from zope.interface.verify import verifyObject
from typing_extensions import NoReturn
from twisted.python import filepath
from twisted.python.filepath import FileMode, OtherAnyStr
from twisted.python.runtime import platform
from twisted.python.win32 import ERROR_DIRECTORY
from twisted.trial.unittest import SynchronousTestCase as TestCase
def setUpFaultyRename(self) -> List[Tuple[str, str]]:
    """
        Set up a C{os.rename} that will fail with L{errno.EXDEV} on first call.
        This is used to simulate a cross-device rename failure.

        @return: a list of pair (src, dest) of calls to C{os.rename}
        @rtype: C{list} of C{tuple}
        """
    invokedWith = []

    def faultyRename(src: str, dest: str) -> None:
        invokedWith.append((src, dest))
        if len(invokedWith) == 1:
            raise OSError(errno.EXDEV, 'Test-induced failure simulating cross-device rename failure')
        return originalRename(src, dest)
    originalRename = os.rename
    self.patch(os, 'rename', faultyRename)
    return invokedWith