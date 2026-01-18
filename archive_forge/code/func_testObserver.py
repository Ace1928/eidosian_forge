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
@implementer(ILogObserver)
def testObserver(e: LogEvent) -> None:
    self.assertIs(e, event)
    self.assertEqual(event['log_trace'], [(publisher, yesFilter), (yesFilter, oYes), (publisher, noFilter), (publisher, oTest)])