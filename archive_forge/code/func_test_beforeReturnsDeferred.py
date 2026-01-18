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
def test_beforeReturnsDeferred(self):
    """
        If a trigger added to the C{'before'} phase of an event returns a
        L{Deferred}, the C{'during'} phase should be delayed until it is called
        back.
        """
    triggerDeferred = Deferred()
    eventType = 'test'
    events = []

    def beforeTrigger():
        return triggerDeferred

    def duringTrigger():
        events.append('during')
    self.addTrigger('before', eventType, beforeTrigger)
    self.addTrigger('during', eventType, duringTrigger)
    self.assertEqual(events, [])
    reactor.fireSystemEvent(eventType)
    self.assertEqual(events, [])
    triggerDeferred.callback(None)
    self.assertEqual(events, ['during'])