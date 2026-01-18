from __future__ import annotations
from zope.interface import verify
from twisted.positioning import base
from twisted.positioning.base import Angles, Directions
from twisted.positioning.ipositioning import IPositioningBeacon
from twisted.trial.unittest import TestCase
def setPDOP(pe: base.PositionError) -> None:
    pe.pdop = 100.0