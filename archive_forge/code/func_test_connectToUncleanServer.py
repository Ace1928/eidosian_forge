import os
import socket
import sys
from unittest import skipIf
from twisted.internet import address, defer, error, interfaces, protocol, reactor, utils
from twisted.python import lockfile
from twisted.python.compat import networkString
from twisted.python.filepath import FilePath
from twisted.test.test_tcp import MyClientFactory, MyServerFactory
from twisted.trial import unittest
def test_connectToUncleanServer(self):
    """
        If passed C{True} for the C{checkPID} parameter, a client connection
        attempt made with L{IReactorUNIX.connectUNIX} fails with
        L{error.BadFileError}.
        """

    def ranStupidChild(ign):
        d = defer.Deferred()
        f = FailedConnectionClientFactory(d)
        reactor.connectUNIX(self.filename, f, checkPID=True)
        return self.assertFailure(d, error.BadFileError)
    return self._uncleanSocketTest(ranStupidChild)