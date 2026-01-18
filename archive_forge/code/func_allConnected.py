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
def allConnected(args):
    serverProtocol, clientProtocol = args
    self.assertEqual(clientFactory.peerAddresses, [address.UNIXAddress(filename)])
    clientProtocol.transport.loseConnection()
    serverProtocol.transport.loseConnection()