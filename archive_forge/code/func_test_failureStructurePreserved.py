from io import BytesIO, StringIO
from typing import IO, Any, List, Optional, Sequence, cast
from zope.interface import implementer
from zope.interface.exceptions import BrokenMethodImplementation
from zope.interface.verify import verifyObject
from twisted.python.failure import Failure
from twisted.trial.unittest import TestCase
from .._flatten import extractField
from .._format import formatEvent
from .._global import globalLogPublisher
from .._interfaces import ILogObserver, LogEvent
from .._json import (
from .._levels import LogLevel
from .._logger import Logger
from .._observer import LogPublisher
def test_failureStructurePreserved(self) -> None:
    """
        Round-tripping a failure through L{eventAsJSON} preserves its class and
        structure.
        """
    events: List[LogEvent] = []
    log = Logger(observer=cast(ILogObserver, events.append))
    try:
        1 / 0
    except ZeroDivisionError:
        f = Failure()
        log.failure('a message about failure', f)
    self.assertEqual(len(events), 1)
    loaded = eventFromJSON(self.savedEventJSON(events[0]))['log_failure']
    self.assertIsInstance(loaded, Failure)
    self.assertTrue(loaded.check(ZeroDivisionError))
    self.assertIsInstance(loaded.getTraceback(), str)