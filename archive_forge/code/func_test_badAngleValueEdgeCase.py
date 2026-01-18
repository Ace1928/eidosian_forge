from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_badAngleValueEdgeCase(self) -> None:
    """
        Headings can not be instantiated with a value of 360 degrees.
        """
    self._badValueTest(angleValue=360.0)