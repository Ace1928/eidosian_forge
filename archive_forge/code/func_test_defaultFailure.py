from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_defaultFailure(self) -> None:
    """
        Test that log.failure() emits the right data.
        """
    log = TestLogger()
    try:
        raise RuntimeError('baloney!')
    except RuntimeError:
        log.failure('Whoops')
    errors = self.flushLoggedErrors(RuntimeError)
    self.assertEqual(len(errors), 1)
    self.assertEqual(log.emitted['level'], LogLevel.critical)
    self.assertEqual(log.emitted['format'], 'Whoops')