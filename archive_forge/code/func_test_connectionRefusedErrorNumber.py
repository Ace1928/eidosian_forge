import errno
import random
import socket
from functools import wraps
from typing import Callable, Optional
from unittest import skipIf
from zope.interface import implementer
import hamcrest
from twisted.internet import defer, error, interfaces, protocol, reactor
from twisted.internet.address import IPv4Address
from twisted.internet.interfaces import IHalfCloseableProtocol, IPullProducer
from twisted.internet.protocol import Protocol
from twisted.internet.testing import AccumulatingProtocol
from twisted.protocols import policies
from twisted.python.log import err, msg
from twisted.python.runtime import platform
from twisted.trial.unittest import SkipTest, TestCase
def test_connectionRefusedErrorNumber(self):
    """
        Assert that the error number of the ConnectionRefusedError is
        ECONNREFUSED, and not some other socket related error.
        """
    serverSockets = []
    for i in range(10):
        serverSocket = socket.socket()
        serverSocket.bind(('127.0.0.1', 0))
        serverSocket.listen(1)
        serverSockets.append(serverSocket)
    random.shuffle(serverSockets)
    clientCreator = protocol.ClientCreator(reactor, protocol.Protocol)

    def tryConnectFailure():

        def connected(proto):
            """
                Darn.  Kill it and try again, if there are any tries left.
                """
            proto.transport.loseConnection()
            if serverSockets:
                return tryConnectFailure()
            self.fail('Could not fail to connect - could not test errno for that case.')
        serverSocket = serverSockets.pop()
        serverHost, serverPort = serverSocket.getsockname()
        serverSocket.close()
        connectDeferred = clientCreator.connectTCP(serverHost, serverPort)
        connectDeferred.addCallback(connected)
        return connectDeferred
    refusedDeferred = tryConnectFailure()
    self.assertFailure(refusedDeferred, error.ConnectionRefusedError)

    def connRefused(exc):
        self.assertEqual(exc.osError, errno.ECONNREFUSED)
    refusedDeferred.addCallback(connRefused)

    def cleanup(passthrough):
        while serverSockets:
            serverSockets.pop().close()
        return passthrough
    refusedDeferred.addBoth(cleanup)
    return refusedDeferred