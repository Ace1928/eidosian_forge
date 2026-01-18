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
def test_badGSAFixType(self) -> None:
    """
        GSA sentence data is not used when the fix claims to be valid
        (albeit only 2D), but the data mode says the data is void.
        Some GPSes do this, unfortunately, and that means you
        shouldn't use the data.
        """
    sentenceData = {'type': 'GPGSA', 'altitude': '545.4', 'dataMode': nmea.GPGLLGPRMCFixQualities.VOID, 'fixType': nmea.GPGSAFixTypes.GSA_2D_FIX}
    self._invalidFixTest(sentenceData)