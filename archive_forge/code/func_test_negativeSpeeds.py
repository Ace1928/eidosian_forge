from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_negativeSpeeds(self) -> None:
    """
        Creating a negative speed raises C{ValueError}.
        """
    self.assertRaises(ValueError, base.Speed, -1.0)