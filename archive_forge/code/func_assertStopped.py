from __future__ import annotations
from typing import Literal
from zope.interface import implementer
from twisted.internet.interfaces import IPushProducer
from twisted.internet.protocol import Protocol
from twisted.internet.task import Clock
from twisted.test.iosim import FakeTransport, connect, connectedServerAndClient
from twisted.trial.unittest import TestCase
def assertStopped(self, producer: StrictPushProducer) -> None:
    """
        Assert that the given producer is in the stopped state.

        @param producer: The producer to verify.
        @type producer: L{StrictPushProducer}
        """
    self.assertEqual(producer._state, 'stopped')