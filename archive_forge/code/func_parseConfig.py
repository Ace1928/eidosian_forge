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
def parseConfig(self, resolvConf):
    servers = []
    for L in resolvConf:
        L = L.strip()
        if L.startswith(b'nameserver'):
            resolver = (nativeString(L.split()[1]), dns.PORT)
            servers.append(resolver)
            log.msg(f'Resolver added {resolver!r} to server list')
        elif L.startswith(b'domain'):
            try:
                self.domain = L.split()[1]
            except IndexError:
                self.domain = b''
            self.search = None
        elif L.startswith(b'search'):
            self.search = L.split()[1:]
            self.domain = None
    if not servers:
        servers.append(('127.0.0.1', dns.PORT))
    self.dynServers = servers