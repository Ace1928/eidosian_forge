from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_headingTooLarge(self) -> None:
    """
        Heading values greater than C{360.0} raise C{ValueError}.
        """
    self._badValueTest(angleValue=370.0)