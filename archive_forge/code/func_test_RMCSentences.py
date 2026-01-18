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
def test_RMCSentences(self) -> None:
    """
        A sequence of RMC sentences fires C{positionReceived},
        C{speedReceived}, C{headingReceived} and C{timeReceived}.
        """
    sentences = [GPRMC]
    callbacksFired = ['headingReceived', 'speedReceived', 'positionReceived', 'timeReceived']
    self._receiverTest(sentences, callbacksFired)