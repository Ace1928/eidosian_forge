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
def testMultiExt(self) -> None:
    f3 = self.path.child(b'sub3').child(b'file3')
    exts = (b'.foo', b'.bar', b'ext1', b'ext2', b'ext3')
    self.assertFalse(f3.siblingExtensionSearch(*exts))
    f3e = f3.siblingExtension(b'.foo')
    f3e.touch()
    found = f3.siblingExtensionSearch(*exts)
    assert found is not None
    self.assertFalse(not found.exists())
    globbed = f3.siblingExtensionSearch(b'*')
    assert globbed is not None
    self.assertFalse(not globbed.exists())
    f3e.remove()
    self.assertFalse(f3.siblingExtensionSearch(*exts))