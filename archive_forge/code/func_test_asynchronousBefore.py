import os
import sys
import time
from unittest import skipIf
from twisted.internet import abstract, base, defer, error, interfaces, protocol, reactor
from twisted.internet.defer import Deferred, passthru
from twisted.internet.tcp import Connector
from twisted.python import util
from twisted.trial.unittest import TestCase
import %(reactor)s
from twisted.internet import reactor
def test_asynchronousBefore(self):
    """
        L{_ThreePhaseEvent.fireEvent} should wait for any L{Deferred} returned
        by a I{before} phase trigger before proceeding to I{during} events.
        """
    events = []
    beforeResult = Deferred()
    self.event.addTrigger('before', lambda: beforeResult)
    self.event.addTrigger('during', events.append, 'during')
    self.event.addTrigger('after', events.append, 'after')
    self.assertEqual(events, [])
    self.event.fireEvent()
    self.assertEqual(events, [])
    beforeResult.callback(None)
    self.assertEqual(events, ['during', 'after'])