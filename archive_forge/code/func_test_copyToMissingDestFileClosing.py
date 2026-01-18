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
def test_copyToMissingDestFileClosing(self) -> None:
    """
        If an exception is raised while L{FilePath.copyTo} is trying to open
        source file to read from, the destination file is closed and the
        exception is raised to the caller of L{FilePath.copyTo}.
        """
    nosuch = self.path.child(b'nothere')
    nosuch.isfile = lambda: True
    destination = ExplodingFilePath(self.mktemp())
    self.assertRaises(IOError, nosuch.copyTo, destination)
    self.assertTrue(destination.fp.closed)