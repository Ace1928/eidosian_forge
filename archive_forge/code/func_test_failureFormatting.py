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
def test_failureFormatting(self) -> None:
    """
        A L{FileLogObserver} created by L{jsonFileLogObserver} writes failures
        serialized as JSON text to a file when it observes events.
        """
    io = StringIO()
    publisher = LogPublisher()
    logged: List[LogEvent] = []
    publisher.addObserver(cast(ILogObserver, logged.append))
    publisher.addObserver(jsonFileLogObserver(io))
    logger = Logger(observer=publisher)
    try:
        1 / 0
    except BaseException:
        logger.failure('failed as expected')
    reader = StringIO(io.getvalue())
    deserialized = list(eventsFromJSONLogFile(reader))

    def checkEvents(logEvents: Sequence[LogEvent]) -> None:
        self.assertEqual(len(logEvents), 1)
        [failureEvent] = logEvents
        self.assertIn('log_failure', failureEvent)
        failureObject = failureEvent['log_failure']
        self.assertIsInstance(failureObject, Failure)
        tracebackObject = failureObject.getTracebackObject()
        self.assertEqual(tracebackObject.tb_frame.f_code.co_filename.rstrip('co'), __file__.rstrip('co'))
    checkEvents(logged)
    checkEvents(deserialized)