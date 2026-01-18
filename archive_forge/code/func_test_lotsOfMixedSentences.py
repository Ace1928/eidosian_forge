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
def test_lotsOfMixedSentences(self) -> None:
    """
        Sends an entire gamut of sentences and verifies the
        appropriate callbacks fire. These are more than you'd expect
        from your average consumer GPS device. They have most of the
        important information, including beacon information and
        visibility.
        """
    sentences = [GPGSA] + GPGSV_SEQ + [GPRMC, GPGGA, GPGLL]
    callbacksFired = ['headingReceived', 'beaconInformationReceived', 'speedReceived', 'positionReceived', 'timeReceived', 'altitudeReceived', 'positionErrorReceived']
    self._receiverTest(sentences, callbacksFired)