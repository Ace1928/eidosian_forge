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
def test_fireEvent(self):
    """
        L{_ThreePhaseEvent.fireEvent} should call I{before}, I{during}, and
        I{after} phase triggers in that order.
        """
    events = []
    self.event.addTrigger('after', events.append, ('first', 'after'))
    self.event.addTrigger('during', events.append, ('first', 'during'))
    self.event.addTrigger('before', events.append, ('first', 'before'))
    self.event.addTrigger('before', events.append, ('second', 'before'))
    self.event.addTrigger('during', events.append, ('second', 'during'))
    self.event.addTrigger('after', events.append, ('second', 'after'))
    self.assertEqual(events, [])
    self.event.fireEvent()
    self.assertEqual(events, [('first', 'before'), ('second', 'before'), ('first', 'during'), ('second', 'during'), ('first', 'after'), ('second', 'after')])