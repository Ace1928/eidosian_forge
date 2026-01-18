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
def logEvent(self, *events: LogEvent) -> Tuple[List[LogRecord], str]:
    """
        Send one or more events to Python's logging module, and capture the
        emitted L{LogRecord}s and output stream as a string.

        @param events: events

        @return: a tuple: (records, output)
        """
    pl = self.py_logger()
    observer = STDLibLogObserver(stackDepth=STDLibLogObserver.defaultStackDepth + 1)
    for event in events:
        observer(event)
    return (pl.bufferedHandler.records, pl.outputAsText())