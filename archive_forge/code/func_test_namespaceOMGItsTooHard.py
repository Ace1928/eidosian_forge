from typing import List, Optional, Type, cast
from zope.interface import implementer
from constantly import NamedConstant
from twisted.trial import unittest
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._levels import InvalidLogLevelError, LogLevel
from .._logger import Logger
def test_namespaceOMGItsTooHard(self) -> None:
    """
        Default namespace is C{"<unknown>"} when a logger is created from a
        context in which is can't be determined automatically and no namespace
        was specified.
        """
    result: List[Logger] = []
    exec('result.append(Logger())', dict(Logger=Logger), locals())
    self.assertEqual(result[0].namespace, '<unknown>')