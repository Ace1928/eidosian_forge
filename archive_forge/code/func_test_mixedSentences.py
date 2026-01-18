from __future__ import annotations
import datetime
from operator import attrgetter
from typing import Callable, Iterable, TypedDict
from zope.interface import implementer
from constantly import NamedConstant
from typing_extensions import Literal, Protocol
from twisted.positioning import base, ipositioning, nmea
from twisted.positioning.base import Angles
from twisted.positioning.test.receiver import MockPositioningReceiver
from twisted.trial.unittest import TestCase
def test_mixedSentences(self) -> None:
    """
        A mix of sentences fires the correct callbacks.
        """
    sentences = [GPRMC, GPGGA]
    callbacksFired = ['altitudeReceived', 'speedReceived', 'positionReceived', 'positionErrorReceived', 'timeReceived', 'headingReceived']

    def checkTime() -> None:
        expectedDateTime = datetime.datetime(1994, 3, 23, 12, 35, 19)
        self.assertEqual(self.adapter._state['time'], expectedDateTime)
    self._receiverTest(sentences, callbacksFired, checkTime)