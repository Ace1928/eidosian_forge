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
def test_finishedBeforeTriggersCleared(self):
    """
        The temporary list L{_ThreePhaseEvent.finishedBefore} should be emptied
        and the state reset to C{'BASE'} before the first during-phase trigger
        executes.
        """
    events = []

    def duringTrigger():
        events.append('during')
        self.assertEqual(self.event.finishedBefore, [])
        self.assertEqual(self.event.state, 'BASE')
    self.event.addTrigger('before', events.append, 'before')
    self.event.addTrigger('during', duringTrigger)
    self.event.fireEvent()
    self.assertEqual(events, ['before', 'during'])