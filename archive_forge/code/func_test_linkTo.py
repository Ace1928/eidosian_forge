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
def test_linkTo(self) -> None:
    """
        Verify that symlink creates a valid symlink that is both a link and a
        file if its target is a file, or a directory if its target is a
        directory.
        """
    targetLinks = [(self.path.child(b'sub2'), self.path.child(b'sub2.link')), (self.path.child(b'sub2').child(b'file3.ext1'), self.path.child(b'file3.ext1.link'))]
    for target, link in targetLinks:
        target.linkTo(link)
        self.assertTrue(link.islink(), 'This is a link')
        self.assertEqual(target.isdir(), link.isdir())
        self.assertEqual(target.isfile(), link.isfile())