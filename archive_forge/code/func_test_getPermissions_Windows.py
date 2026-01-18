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
@skipIf(not platform.isWindows(), 'Test will run only on Windows')
def test_getPermissions_Windows(self) -> None:
    """
        Getting permissions for a file returns a L{Permissions} object in
        Windows.  Windows requires a different test, because user permissions
        = group permissions = other permissions.  Also, chmod may not be able
        to set the execute bit, so we are skipping tests that set the execute
        bit.
        """
    self.addCleanup(self.path.child(b'sub1').chmod, 511)
    for mode in (511, 365):
        self.path.child(b'sub1').chmod(mode)
        self.assertEqual(self.path.child(b'sub1').getPermissions(), filepath.Permissions(mode))
    self.path.child(b'sub1').chmod(329)
    self.assertEqual(self.path.child(b'sub1').getPermissions().shorthand(), 'r-xr-xr-x')