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
def test_mixedTypeGlobChildren(self) -> None:
    """
        C{globChildren} will return the same type as the pattern argument.
        """
    fp = filepath.FilePath('/')
    children = fp.globChildren(b'*')
    self.assertIsInstance(children[0].path, bytes)