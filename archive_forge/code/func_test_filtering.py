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
def test_filtering(self) -> None:
    """
        Events are filtered based on log level/namespace.
        """
    predicate = LogLevelFilterPredicate()
    predicate.setLogLevelForNamespace('', LogLevel.error)
    predicate.setLogLevelForNamespace('twext.web2', LogLevel.debug)
    predicate.setLogLevelForNamespace('twext.web2.dav', LogLevel.warn)

    def checkPredicate(namespace: str, level: NamedConstant, expectedResult: NamedConstant) -> None:
        event: LogEvent = dict(log_namespace=namespace, log_level=level)
        self.assertEqual(expectedResult, predicate(event))
    checkPredicate('', LogLevel.debug, PredicateResult.no)
    checkPredicate(cast(str, None), LogLevel.debug, PredicateResult.no)
    checkPredicate('', LogLevel.error, PredicateResult.no)
    checkPredicate(cast(str, None), LogLevel.error, PredicateResult.no)
    checkPredicate('twext.web2', LogLevel.debug, PredicateResult.maybe)
    checkPredicate('twext.web2', LogLevel.error, PredicateResult.maybe)
    checkPredicate('twext.web2.dav', LogLevel.debug, PredicateResult.no)
    checkPredicate('twext.web2.dav', LogLevel.error, PredicateResult.maybe)
    checkPredicate('', LogLevel.critical, PredicateResult.no)
    checkPredicate(cast(str, None), LogLevel.critical, PredicateResult.no)
    checkPredicate('twext.web2', None, PredicateResult.no)