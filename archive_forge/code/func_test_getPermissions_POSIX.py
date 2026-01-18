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
@skipIf(platform.isWindows(), 'Test does not run on Windows')
def test_getPermissions_POSIX(self) -> None:
    """
        Getting permissions for a file returns a L{Permissions} object for
        POSIX platforms (which supports separate user, group, and other
        permissions bits.
        """
    for mode in (511, 448):
        self.path.child(b'sub1').chmod(mode)
        self.assertEqual(self.path.child(b'sub1').getPermissions(), filepath.Permissions(mode))
    self.path.child(b'sub1').chmod(500)
    self.assertEqual(self.path.child(b'sub1').getPermissions().shorthand(), 'rwxrw-r--')