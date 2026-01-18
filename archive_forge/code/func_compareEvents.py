import io
from typing import IO, Any, List, Optional, TextIO, Tuple, Type, cast
from twisted.python.failure import Failure
from twisted.trial import unittest
from .._file import textFileLogObserver
from .._global import MORE_THAN_ONCE_WARNING, LogBeginner
from .._interfaces import ILogObserver, LogEvent
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
from ..test.test_stdlib import nextLine
def compareEvents(test: unittest.TestCase, actualEvents: List[LogEvent], expectedEvents: List[LogEvent]) -> None:
    """
    Compare two sequences of log events, examining only the the keys which are
    present in both.

    @param test: a test case doing the comparison
    @param actualEvents: A list of log events that were emitted by a logger.
    @param expectedEvents: A list of log events that were expected by a test.
    """
    if len(actualEvents) != len(expectedEvents):
        test.assertEqual(actualEvents, expectedEvents)
    allMergedKeys = set()
    for event in expectedEvents:
        allMergedKeys |= set(event.keys())

    def simplify(event: LogEvent) -> LogEvent:
        copy = event.copy()
        for key in event.keys():
            if key not in allMergedKeys:
                copy.pop(key)
        return copy
    simplifiedActual = [simplify(event) for event in actualEvents]
    test.assertEqual(simplifiedActual, expectedEvents)