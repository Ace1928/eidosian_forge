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
def test_makedirsRaisesNonEexistErrorsIgnoreExistingDirectory(self) -> None:
    """
        When C{FilePath.makedirs} is called with C{ignoreExistingDirectory} set
        to C{True} it raises an C{OSError} exception if exception errno is not
        EEXIST.
        """

    def faultyMakedirs(path: str) -> None:
        raise OSError(errno.EACCES, 'Permission Denied')
    self.patch(os, 'makedirs', faultyMakedirs)
    fp = filepath.FilePath(self.mktemp())
    exception = self.assertRaises(OSError, fp.makedirs, ignoreExistingDirectory=True)
    self.assertEqual(exception.errno, errno.EACCES)