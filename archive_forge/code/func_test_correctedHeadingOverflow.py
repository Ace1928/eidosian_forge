from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def test_correctedHeadingOverflow(self) -> None:
    """
        A heading with a value and a variation has the appropriate corrected
        heading value, even when the variation puts it across the 360 degree
        boundary.
        """
    h = base.Heading.fromFloats(359.0, variationValue=-2.0)
    self.assertEqual(h.correctedHeading, base.Angle(1.0, Angles.HEADING))