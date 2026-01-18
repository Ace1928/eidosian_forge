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
def test_callbacksCalled(self) -> None:
    """
        The correct callbacks fire, and that *only* those fire.
        """
    sentencesByType = {'GPGGA': [b'$GPGGA*56'], 'GPGLL': [b'$GPGLL*50'], 'GPGSA': [b'$GPGSA*42'], 'GPGSV': [b'$GPGSV*55'], 'GPHDT': [b'$GPHDT*4f'], 'GPRMC': [b'$GPRMC*4b']}
    for sentenceType, sentences in sentencesByType.items():
        for sentence in sentences:
            self.protocol.lineReceived(sentence)
            self.assertEqual(self.sentenceTypes, {sentenceType})
            self.sentenceTypes.clear()