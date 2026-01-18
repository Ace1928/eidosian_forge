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
def test_nameCollision(self) -> None:
    """
        L{FilePath.setContent} will use a different temporary filename on each
        invocation, so that multiple processes, threads, or reentrant
        invocations will not collide with each other.
        """
    fp = TrackingFilePath(self.mktemp())
    fp.setContent(b'alpha')
    fp.setContent(b'beta')
    openedSiblings = fp.openedPaths()
    self.assertEqual(len(openedSiblings), 2)
    self.assertNotEqual(openedSiblings[0], openedSiblings[1])