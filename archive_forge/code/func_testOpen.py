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
def testOpen(self) -> None:
    nonexistent = self.path.child(b'nonexistent')
    e = self.assertRaises(IOError, nonexistent.open)
    self.assertEqual(e.errno, errno.ENOENT)
    writer = self.path.child(b'writer')
    with writer.open('w') as f:
        f.write(b'abc\ndef')
    with writer.open() as f:
        self.assertEqual(f.read(), b'abc\ndef')
    writer.open('w').close()
    with writer.open() as f:
        self.assertEqual(f.read(), b'')
    appender = self.path.child(b'appender')
    with appender.open('w') as f:
        f.write(b'abc')
    with appender.open('a') as f:
        f.write(b'def')
    with appender.open('r') as f:
        self.assertEqual(f.read(), b'abcdef')
    with appender.open('r+') as f:
        self.assertEqual(f.read(), b'abcdef')
        f.seek(0, 1)
        f.write(b'ghi')
    with appender.open('r') as f:
        self.assertEqual(f.read(), b'abcdefghi')
    with appender.open('w+') as f:
        self.assertEqual(f.read(), b'')
        f.seek(0, 1)
        f.write(b'123')
    with appender.open('a+') as f:
        f.write(b'456')
        f.seek(0, 1)
        self.assertEqual(f.read(), b'')
        f.seek(0, 0)
        self.assertEqual(f.read(), b'123456')
    nonexistent.requireCreate(True)
    nonexistent.open('w').close()
    existent = nonexistent
    del nonexistent
    self.assertRaises((OSError, IOError), existent.open)