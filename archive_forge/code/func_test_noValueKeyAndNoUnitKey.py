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
def test_noValueKeyAndNoUnitKey(self) -> None:
    """
        Tests that when a unit is specified but neither C{valueKey} nor
        C{unitKey} is provided, C{ValueError} is raised.
        """
    self.assertRaises(ValueError, self.adapter._fixUnits, unit='K')