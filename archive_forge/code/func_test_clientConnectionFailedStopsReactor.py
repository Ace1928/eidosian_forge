import socket
from gc import collect
from typing import Optional
from weakref import ref
from zope.interface.verify import verifyObject
from twisted.internet.defer import Deferred, gatherResults
from twisted.internet.interfaces import IConnector, IReactorFDSet
from twisted.internet.protocol import ClientFactory, Protocol, ServerFactory
from twisted.internet.test.reactormixins import needsRunningReactor
from twisted.python import context, log
from twisted.python.failure import Failure
from twisted.python.log import ILogContext, err, msg
from twisted.python.runtime import platform
from twisted.test.test_tcp import ClosingProtocol
from twisted.trial.unittest import SkipTest
def test_clientConnectionFailedStopsReactor(self):
    """
        The reactor can be stopped by a client factory's
        C{clientConnectionFailed} method.
        """
    reactor = self.buildReactor()
    needsRunningReactor(reactor, lambda: self.connect(reactor, Stop(reactor)))
    self.runReactor(reactor)