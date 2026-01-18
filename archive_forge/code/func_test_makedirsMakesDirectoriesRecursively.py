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
def test_makedirsMakesDirectoriesRecursively(self) -> None:
    """
        C{FilePath.makedirs} creates a directory at C{path}}, including
        recursively creating all parent directories leading up to the path.
        """
    fp = filepath.FilePath(os.path.join(self.mktemp(), b'foo', b'bar', b'baz'))
    self.assertFalse(fp.exists())
    fp.makedirs()
    self.assertTrue(fp.exists())
    self.assertTrue(fp.isdir())