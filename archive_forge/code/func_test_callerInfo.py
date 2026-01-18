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
def test_callerInfo(self) -> None:
    """
        C{pathname}, C{lineno}, C{exc_info}, C{func} is set properly on
        records.
        """
    filename, logLine = nextLine()
    records, output = self.logEvent({})
    self.assertEqual(len(records), 1)
    self.assertEqual(records[0].pathname, filename)
    self.assertEqual(records[0].lineno, logLine)
    self.assertIsNone(records[0].exc_info)