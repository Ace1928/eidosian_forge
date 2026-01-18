import pickle
from zope.interface import implementer
from zope.interface.verify import verifyClass
from twisted.application import internet
from twisted.application.internet import (
from twisted.internet import task
from twisted.internet.defer import CancelledError, Deferred
from twisted.internet.interfaces import (
from twisted.internet.protocol import Factory, Protocol
from twisted.internet.task import Clock
from twisted.internet.testing import StringTransport
from twisted.logger import formatEvent, globalLogPublisher
from twisted.python.failure import Failure
from twisted.trial.unittest import SynchronousTestCase, TestCase
def makeReconnector(self, fireImmediately=True, startService=True, protocolType=Protocol, **kw):
    """
        Create a L{ClientService} along with a L{ConnectInformation} indicating
        the connections in progress on its endpoint.

        @param fireImmediately: Should all of the endpoint connection attempts
            fire synchronously?
        @type fireImmediately: L{bool}

        @param startService: Should the L{ClientService} be started before
            being returned?
        @type startService: L{bool}

        @param protocolType: a 0-argument callable returning a new L{IProtocol}
            provider to be used for application-level protocol connections.

        @param kw: Arbitrary keyword arguments to be passed on to
            L{ClientService}

        @return: a 2-tuple of L{ConnectInformation} (for information about test
            state) and L{ClientService} (the system under test).  The
            L{ConnectInformation} has 2 additional attributes;
            C{applicationFactory} and C{applicationProtocols}, which refer to
            the unwrapped protocol factory and protocol instances passed in to
            L{ClientService} respectively.
        """
    nkw = {}
    nkw.update(clock=Clock())
    nkw.update(kw)
    clock = nkw['clock']
    cq, endpoint = endpointForTesting(fireImmediately=fireImmediately)
    applicationProtocols = cq.applicationProtocols = []

    class RememberingFactory(Factory):
        protocol = protocolType

        def buildProtocol(self, addr):
            result = super().buildProtocol(addr)
            applicationProtocols.append(result)
            return result
    cq.applicationFactory = factory = RememberingFactory()
    service = ClientService(endpoint, factory, **nkw)

    def stop():
        service._protocol = None
        if service.running:
            service.stopService()
        self.assertEqual(clock.getDelayedCalls(), [])
    self.addCleanup(stop)
    if startService:
        service.startService()
    return (cq, service)