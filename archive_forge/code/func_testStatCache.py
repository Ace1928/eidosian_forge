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
def testStatCache(self) -> None:
    p = self.path.child(b'stattest')
    p.touch()
    self.assertEqual(p.getsize(), 0)
    self.assertEqual(abs(p.getmtime() - time.time()) // 20, 0)
    self.assertEqual(abs(p.getctime() - time.time()) // 20, 0)
    self.assertEqual(abs(p.getatime() - time.time()) // 20, 0)
    self.assertTrue(p.exists())
    self.assertTrue(p.exists())
    os.remove(p.path)
    self.assertTrue(p.exists())
    p.restat(reraise=False)
    self.assertFalse(p.exists())
    self.assertFalse(p.islink())
    self.assertFalse(p.isdir())
    self.assertFalse(p.isfile())