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
def test_duringPreceedsAfter(self):
    """
        L{IReactorCore.addSystemEventTrigger} should call triggers added to the
        C{'during'} phase before it calls triggers added to the C{'after'}
        phase.
        """
    eventType = 'test'
    events = []

    def duringTrigger():
        events.append('during')

    def afterTrigger():
        events.append('after')
    self.addTrigger('during', eventType, duringTrigger)
    self.addTrigger('after', eventType, afterTrigger)
    self.assertEqual(events, [])
    reactor.fireSystemEvent(eventType)
    self.assertEqual(events, ['during', 'after'])