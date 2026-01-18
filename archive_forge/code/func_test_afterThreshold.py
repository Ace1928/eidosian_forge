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
def test_afterThreshold(self) -> None:
    """
        Dates after the threshold are interpreted as being in the same century
        as the threshold.
        """
    datestring, date = ('010195', datetime.date(1995, 1, 1))
    self._fixerTest({'datestamp': datestring}, {'_date': date})