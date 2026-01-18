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
def test_cleanedFailure(self) -> None:
    """
        A cleaned Failure object has a fake traceback object; make sure that
        logging such a failure still results in the exception details being
        logged.
        """

    def failing_func() -> None:
        1 / 0
    try:
        failing_func()
    except ZeroDivisionError:
        failure = Failure()
        failure.cleanFailure()
    event = dict(log_format='Hi mom', who='me', log_failure=failure)
    records, output = self.logEvent(event)
    self.assertEqual(len(records), 1)
    self.assertIn('Hi mom', output)
    self.assertIn('in failing_func', output)
    self.assertIn('ZeroDivisionError', output)