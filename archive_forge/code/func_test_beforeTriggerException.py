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
def test_beforeTriggerException(self):
    """
        If a before-phase trigger raises a synchronous exception, it should be
        logged and the remaining triggers should be run.
        """
    events = []

    class DummyException(Exception):
        pass

    def raisingTrigger():
        raise DummyException()
    self.event.addTrigger('before', raisingTrigger)
    self.event.addTrigger('before', events.append, 'before')
    self.event.addTrigger('during', events.append, 'during')
    self.event.fireEvent()
    self.assertEqual(events, ['before', 'during'])
    errors = self.flushLoggedErrors(DummyException)
    self.assertEqual(len(errors), 1)