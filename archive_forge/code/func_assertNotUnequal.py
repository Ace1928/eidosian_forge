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
def assertNotUnequal(self, first: T, second: object, msg: Optional[str]=None) -> T:
    """
        Tests that C{first} != C{second} is false.  This method tests the
        __ne__ method, as opposed to L{assertEqual} (C{first} == C{second}),
        which tests the __eq__ method.

        Note: this should really be part of trial
        """
    if first != second:
        if msg is None:
            msg = ''
        if len(msg) > 0:
            msg += '\n'
        raise self.failureException('%snot not unequal (__ne__ not implemented correctly):\na = %s\nb = %s\n' % (msg, pformat(first), pformat(second)))
    return first