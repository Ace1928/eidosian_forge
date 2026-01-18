import errno
import os
import warnings
from zope.interface import moduleProvides
from twisted.internet import defer, error, interfaces, protocol
from twisted.internet.abstract import isIPv6Address
from twisted.names import cache, common, dns, hosts as hostsModule, resolve, root
from twisted.python import failure, log
from twisted.python.compat import nativeString
from twisted.python.filepath import FilePath
from twisted.python.runtime import platform
from twisted.internet.base import ThreadedResolver as _ThreadedResolverImpl
def pickServer(self):
    """
        Return the address of a nameserver.

        TODO: Weight servers for response time so faster ones can be
        preferred.
        """
    if not self.servers and (not self.dynServers):
        return None
    serverL = len(self.servers)
    dynL = len(self.dynServers)
    self.index += 1
    self.index %= serverL + dynL
    if self.index < serverL:
        return self.servers[self.index]
    else:
        return self.dynServers[self.index - serverL]