from __future__ import annotations
import gc
from typing import Union
from zope.interface import Interface, directlyProvides, implementer
from zope.interface.verify import verifyObject
from hypothesis import given, strategies as st
from twisted.internet import reactor
from twisted.internet.task import Clock, deferLater
from twisted.python.compat import iterbytes
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.error import ConnectionDone, ConnectionLost
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, Protocol, ServerFactory
from twisted.internet.task import TaskStopped
from twisted.internet.testing import NonStreamingProducer, StringTransport
from twisted.protocols.loopback import collapsingPumpPolicy, loopbackAsync
from twisted.python import log
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.test.iosim import connectedServerAndClient
from twisted.test.test_tcp import ConnectionLostNotifyingProtocol
from twisted.trial.unittest import SynchronousTestCase, TestCase
def drain(self, transport, allowEmpty=False):
    """
        Drain the bytes currently pending write from a L{StringTransport}, then
        clear it, since those bytes have been consumed.

        @param transport: The L{StringTransport} to get the bytes from.
        @type transport: L{StringTransport}

        @param allowEmpty: Allow the test to pass even if the transport has no
            outgoing bytes in it.
        @type allowEmpty: L{bool}

        @return: the outgoing bytes from the given transport
        @rtype: L{bytes}
        """
    value = transport.value()
    transport.clear()
    self.assertEqual(bool(allowEmpty or value), True)
    return value