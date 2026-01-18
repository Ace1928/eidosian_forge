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
def test_fullGGA(self) -> None:
    """
        A full GGA sentence is correctly parsed.
        """
    expected = {'type': 'GPGGA', 'altitude': '545.4', 'altitudeUnits': 'M', 'heightOfGeoidAboveWGS84': '46.9', 'heightOfGeoidAboveWGS84Units': 'M', 'horizontalDilutionOfPrecision': '0.9', 'latitudeFloat': '4807.038', 'latitudeHemisphere': 'N', 'longitudeFloat': '01131.000', 'longitudeHemisphere': 'E', 'numberOfSatellitesSeen': '08', 'timestamp': '123519', 'fixQuality': '1'}
    self._parserTest(GPGGA, expected)