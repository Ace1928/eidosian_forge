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
def test_fullGSV(self) -> None:
    """
        A full GSV sentence is correctly parsed.
        """
    expected = {'type': 'GPGSV', 'GSVSentenceIndex': '1', 'numberOfGSVSentences': '3', 'numberOfSatellitesSeen': '11', 'azimuth_0': '111', 'azimuth_1': '270', 'azimuth_2': '010', 'azimuth_3': '292', 'elevation_0': '03', 'elevation_1': '15', 'elevation_2': '01', 'elevation_3': '06', 'satellitePRN_0': '03', 'satellitePRN_1': '04', 'satellitePRN_2': '06', 'satellitePRN_3': '13', 'signalToNoiseRatio_0': '00', 'signalToNoiseRatio_1': '00', 'signalToNoiseRatio_2': '00', 'signalToNoiseRatio_3': '00'}
    self._parserTest(GPGSV_FIRST, expected)