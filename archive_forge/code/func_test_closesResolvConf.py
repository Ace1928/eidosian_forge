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
def test_closesResolvConf(self):
    """
        As part of its constructor, C{StubResolver} opens C{/etc/resolv.conf};
        then, explicitly closes it and does not count on the GC to do so for
        it.
        """
    handle = FilePath(self.mktemp())
    resolvConf = handle.open(mode='w+')

    class StubResolver(client.Resolver):

        def _openFile(self, name):
            return resolvConf
    StubResolver(servers=['example.com', 53], resolv='/etc/resolv.conf', reactor=Clock())
    self.assertTrue(resolvConf.closed)