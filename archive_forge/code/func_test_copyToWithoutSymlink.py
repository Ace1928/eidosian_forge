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
@skipIf(symlinkSkip, 'Platform does not support symlinks')
def test_copyToWithoutSymlink(self) -> None:
    """
        Verify that copying with followLinks=False copies symlinks as symlinks
        """
    os.symlink(b'sub1', self.path.child(b'link1').path)
    fp = filepath.FilePath(self.mktemp())
    self.path.copyTo(fp, followLinks=False)
    self.assertTrue(fp.child(b'link1').islink())
    self.assertEqual(os.readlink(self.path.child(b'link1').path), os.readlink(fp.child(b'link1').path))