from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_negativeLongitude(self) -> None:
    """
        Negative longitudes have a repr that specifies their type and value.
        """
    longitude = base.Coordinate(-50.0, Angles.LONGITUDE)
    expectedRepr = f'<Longitude ({-50.0} degrees)>'
    self.assertEqual(repr(longitude), expectedRepr)