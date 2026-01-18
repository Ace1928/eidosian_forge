import errno
from zope.interface.verify import verifyClass, verifyObject
from twisted.internet import defer
from twisted.internet.error import CannotListenError, ConnectionRefusedError
from twisted.internet.interfaces import IResolver
from twisted.internet.task import Clock
from twisted.internet.test.modulehelpers import AlternateReactor
from twisted.names import cache, client, dns, error, hosts
from twisted.names.common import ResolverBase
from twisted.names.error import DNSQueryTimeoutError
from twisted.names.test import test_util
from twisted.names.test.test_hosts import GoodTempPathMixin
from twisted.python import failure
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.test import proto_helpers
from twisted.trial import unittest
def test_connectedProtocol(self):
    """
        L{client.Resolver._connectedProtocol} returns a new
        L{DNSDatagramProtocol} connected to a new address with a
        cryptographically secure random port number.
        """
    resolver = client.Resolver(servers=[('example.com', 53)])
    firstProto = resolver._connectedProtocol()
    secondProto = resolver._connectedProtocol()
    self.assertIsNotNone(firstProto.transport)
    self.assertIsNotNone(secondProto.transport)
    self.assertNotEqual(firstProto.transport.getHost().port, secondProto.transport.getHost().port)
    return defer.gatherResults([defer.maybeDeferred(firstProto.transport.stopListening), defer.maybeDeferred(secondProto.transport.stopListening)])