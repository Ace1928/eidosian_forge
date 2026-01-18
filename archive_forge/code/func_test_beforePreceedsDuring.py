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
def test_beforePreceedsDuring(self):
    """
        L{IReactorCore.addSystemEventTrigger} should call triggers added to the
        C{'before'} phase before it calls triggers added to the C{'during'}
        phase.
        """
    eventType = 'test'
    events = []

    def beforeTrigger():
        events.append('before')

    def duringTrigger():
        events.append('during')
    self.addTrigger('before', eventType, beforeTrigger)
    self.addTrigger('during', eventType, duringTrigger)
    self.assertEqual(events, [])
    reactor.fireSystemEvent(eventType)
    self.assertEqual(events, ['before', 'during'])