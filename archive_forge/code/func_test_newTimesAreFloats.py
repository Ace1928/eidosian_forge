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
def test_newTimesAreFloats(self) -> None:
    """
        Verify that all times returned from the various new time functions are ints
        (and hopefully therefore 'high precision').
        """
    for p in (self.path, self.path.child(b'file1')):
        self.assertEqual(type(p.getAccessTime()), float)
        self.assertEqual(type(p.getModificationTime()), float)
        self.assertEqual(type(p.getStatusChangeTime()), float)