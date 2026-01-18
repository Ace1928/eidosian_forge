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
def test_copyToFileClosing(self) -> None:
    """
        If an exception is raised while L{FilePath.copyTo} is copying bytes
        between two regular files, the source and destination files are closed
        and the exception propagates to the caller of L{FilePath.copyTo}.
        """
    destination = ExplodingFilePath(self.mktemp())
    source = ExplodingFilePath(__file__)
    self.assertRaises(IOError, source.copyTo, destination)
    self.assertTrue(source.fp.closed)
    self.assertTrue(destination.fp.closed)