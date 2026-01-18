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
def test_asynchronousBeforeRemovesDuring(self):
    """
        If a before-phase trigger returns a L{Deferred} and later removes a
        during-phase trigger before the L{Deferred} fires, the during-phase
        trigger should not be run.
        """
    events = []
    beforeResult = Deferred()
    self.event.addTrigger('before', lambda: beforeResult)
    duringHandle = self.event.addTrigger('during', events.append, 'during')
    self.event.addTrigger('after', events.append, 'after')
    self.event.fireEvent()
    self.event.removeTrigger(duringHandle)
    beforeResult.callback(None)
    self.assertEqual(events, ['after'])