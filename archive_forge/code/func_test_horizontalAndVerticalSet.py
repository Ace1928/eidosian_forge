from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_horizontalAndVerticalSet(self) -> None:
    """
        The PDOP is correctly determined from HDOP and VDOP.
        """
    hdop, vdop = (1.0, 1.0)
    pdop = (hdop ** 2 + vdop ** 2) ** 0.5
    pe = base.PositionError(hdop=hdop, vdop=vdop)
    self._testDOP(pe, pdop, hdop, vdop)