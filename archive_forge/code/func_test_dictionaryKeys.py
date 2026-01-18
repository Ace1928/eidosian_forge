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
def test_dictionaryKeys(self) -> None:
    """
        Verify that path instances are usable as dictionary keys.
        """
    f1 = self.path.child(b'file1')
    f1prime = self.path.child(b'file1')
    f2 = self.path.child(b'file2')
    dictoid = {}
    dictoid[f1] = 3
    dictoid[f1prime] = 4
    self.assertEqual(dictoid[f1], 4)
    self.assertEqual(list(dictoid.keys()), [f1])
    self.assertIs(list(dictoid.keys())[0], f1)
    self.assertIsNot(list(dictoid.keys())[0], f1prime)
    dictoid[f2] = 5
    self.assertEqual(dictoid[f2], 5)
    self.assertEqual(len(dictoid), 2)