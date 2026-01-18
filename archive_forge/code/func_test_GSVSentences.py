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
def test_GSVSentences(self) -> None:
    """
        A complete sequence of GSV sentences fires
        C{beaconInformationReceived}.
        """
    sentences = [GPGSV_FIRST, GPGSV_MIDDLE, GPGSV_LAST]
    callbacksFired = ['beaconInformationReceived']

    def checkPartialInformation() -> None:
        self.assertNotIn('_partialBeaconInformation', self.adapter._state)
    self._receiverTest(sentences, callbacksFired, checkPartialInformation)