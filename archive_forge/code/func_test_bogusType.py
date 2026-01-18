from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_bogusType(self) -> None:
    """
        Trying to create an angle with a bogus type raises C{ValueError}.
        """
    self.assertRaises(ValueError, base.Angle, angleType='BOGUS')