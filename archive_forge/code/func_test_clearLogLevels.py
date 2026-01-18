from typing import Iterable, List, Tuple, Union, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from constantly import NamedConstant
from twisted.trial import unittest
from .._filter import (
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._observer import LogPublisher, bitbucketLogObserver
def test_clearLogLevels(self) -> None:
    """
        Clearing log levels.
        """
    predicate = LogLevelFilterPredicate()
    predicate.setLogLevelForNamespace('twext.web2', LogLevel.debug)
    predicate.setLogLevelForNamespace('twext.web2.dav', LogLevel.error)
    predicate.clearLogLevels()
    self.assertEqual(predicate.logLevelForNamespace('twisted'), predicate.defaultLogLevel)
    self.assertEqual(predicate.logLevelForNamespace('twext.web2'), predicate.defaultLogLevel)
    self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav'), predicate.defaultLogLevel)
    self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav.test'), predicate.defaultLogLevel)
    self.assertEqual(predicate.logLevelForNamespace('twext.web2.dav.test1.test2'), predicate.defaultLogLevel)