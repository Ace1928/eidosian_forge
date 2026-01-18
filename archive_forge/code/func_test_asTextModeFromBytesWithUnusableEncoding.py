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
def test_asTextModeFromBytesWithUnusableEncoding(self) -> None:
    """
        C{asTextMode} with an C{encoding} argument that can't be used to encode
        the unicode path raises a L{UnicodeError}.
        """
    fp = filepath.FilePath(b'\\u2603')
    with self.assertRaises(UnicodeError):
        fp.asTextMode(encoding='utf-32')