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
def assertObserverWritesJSON(self, recordSeparator: str='\x1e') -> None:
    """
        Asserts that an observer created by L{jsonFileLogObserver} with the
        given arguments writes events serialized as JSON text, using the given
        record separator.

        @param recordSeparator: C{recordSeparator} argument to
            L{jsonFileLogObserver}
        """
    with StringIO() as fileHandle:
        observer = jsonFileLogObserver(fileHandle, recordSeparator)
        event = dict(x=1)
        observer(event)
        self.assertEqual(fileHandle.getvalue(), f'{recordSeparator}{{"x": 1}}\n')