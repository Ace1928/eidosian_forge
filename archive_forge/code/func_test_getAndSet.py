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
def test_getAndSet(self) -> None:
    content = b'newcontent'
    self.path.child(b'new').setContent(content)
    newcontent = self.path.child(b'new').getContent()
    self.assertEqual(content, newcontent)
    content = b'content'
    self.path.child(b'new').setContent(content, b'.tmp')
    newcontent = self.path.child(b'new').getContent()
    self.assertEqual(content, newcontent)