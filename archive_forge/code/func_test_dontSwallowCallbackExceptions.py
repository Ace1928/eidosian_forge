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
def test_dontSwallowCallbackExceptions(self) -> None:
    """
        An C{AttributeError} in the sentence callback of an C{NMEAProtocol}
        doesn't get swallowed.
        """
    lineReceived = self.protocol.lineReceived
    self.assertRaises(AttributeError, lineReceived, b'$GPGGA*56')