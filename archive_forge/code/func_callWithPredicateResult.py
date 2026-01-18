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
def callWithPredicateResult(result: NamedConstant) -> List[LogEvent]:
    seen: List[LogEvent] = []
    observer = FilteringLogObserver(cast(ILogObserver, lambda e: seen.append(e)), (cast(ILogFilterPredicate, lambda e: result),))
    observer(e)
    return seen