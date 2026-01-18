from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_conflictingKwargs(self) -> None:
    """
        Make sure that kwargs conflicting with args don't pass through.
        """
    log = TestLogger()
    log.warn('*', log_format='#', log_level=LogLevel.error, log_namespace='*namespace*', log_source='*source*')
    self.assertEqual(log.event['log_format'], '*')
    self.assertEqual(log.event['log_level'], LogLevel.warn)
    self.assertEqual(log.event['log_namespace'], log.namespace)
    self.assertIsNone(log.event['log_source'])