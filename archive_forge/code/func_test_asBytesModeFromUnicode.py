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
def test_asBytesModeFromUnicode(self) -> None:
    """
        C{asBytesMode} on a L{unicode}-mode L{FilePath} returns a new
        L{bytes}-mode L{FilePath}.
        """
    fp = filepath.FilePath('./tmp')
    newfp = fp.asBytesMode()
    self.assertIsNot(fp, newfp)
    self.assertIsInstance(newfp.path, bytes)