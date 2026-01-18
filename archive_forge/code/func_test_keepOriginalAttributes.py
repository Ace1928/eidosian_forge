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
def test_keepOriginalAttributes(self) -> None:
    """
        Verify that the Unlistable exception raised will preserve the attributes of
        the previously-raised exception.
        """
    fp = filepath.FilePath(self.mktemp())
    ose = self.assertRaises(OSError, fp.children)
    d1 = list(ose.__dict__.keys())
    d1.remove('originalException')
    d2 = list(ose.originalException.__dict__.keys())
    d1.sort()
    d2.sort()
    self.assertEqual(d1, d2)