from __future__ import annotations
import logging as py_logging
import sys
from inspect import getsourcefile
from io import BytesIO, TextIOWrapper
from logging import Formatter, LogRecord, StreamHandler, getLogger
from typing import List, Optional, Tuple
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.compat import currentframe
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._stdlib import STDLibLogObserver
def test_basicFormat(self) -> None:
    """
        Basic formattable event passes the format along correctly.
        """
    event = dict(log_format='Hello, {who}!', who='dude')
    records, output = self.logEvent(event)
    self.assertEqual(len(records), 1)
    self.assertEqual(str(records[0].msg), 'Hello, dude!')
    self.assertEqual(records[0].args, ())