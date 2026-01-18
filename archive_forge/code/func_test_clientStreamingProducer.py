from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def test_clientStreamingProducer(self) -> None:
    """
        L{IOPump.pump} does not call C{resumeProducing} on a L{IPushProducer}
        (stream producer) registered with the client transport.
        """
    self._testStreamingProducer(mode='client')