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
def test_makedirsMakesDirectoriesWithIgnoreExistingDirectory(self) -> None:
    """
        Calling C{FilePath.makedirs} with C{ignoreExistingDirectory} set to
        C{True} has no effect if directory does not exist.
        """
    fp = filepath.FilePath(self.mktemp())
    self.assertFalse(fp.exists())
    fp.makedirs(ignoreExistingDirectory=True)
    self.assertTrue(fp.exists())
    self.assertTrue(fp.isdir())