from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_asFloat(self) -> None:
    """
        A climb can be converted into a C{float}.
        """
    self.assertEqual(1.0, float(base.Climb(1.0)))