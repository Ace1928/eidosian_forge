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
def test_GLL(self) -> None:
    """
        GLL sentence data is unused when the data is flagged as void.
        """
    sentenceData = {'type': 'GPGLL', 'altitude': '545.4', 'dataMode': nmea.GPGLLGPRMCFixQualities.VOID}
    self._invalidFixTest(sentenceData)