import os
from twisted.internet.defer import Deferred
from twisted.internet.interfaces import IReadDescriptor
from twisted.internet.posixbase import PosixReactorBase, _Waker
from twisted.internet.protocol import ServerFactory
from twisted.python.runtime import platform
from twisted.trial.unittest import TestCase
from twisted.internet import reactor
from twisted.internet.tcp import Port
def test_connectionLostFailed(self):
    """
        L{Port.stopListening} returns a L{Deferred} which errbacks if
        L{Port.connectionLost} raises an exception.
        """
    port = Port(12345, ServerFactory())
    port.connected = True
    port.connectionLost = lambda reason: 1 // 0
    return self.assertFailure(port.stopListening(), ZeroDivisionError)