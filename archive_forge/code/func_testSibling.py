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
def testSibling(self) -> None:
    p = self.path.child(b'sibling_start')
    ts = p.sibling(b'sibling_test')
    self.assertEqual(ts.dirname(), p.dirname())
    self.assertEqual(ts.basename(), b'sibling_test')
    ts.createDirectory()
    self.assertIn(ts, self.path.children())