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
def getResolver():
    """
    Get a Resolver instance.

    Create twisted.names.client.theResolver if it is L{None}, and then return
    that value.

    @rtype: C{IResolver}
    """
    global theResolver
    if theResolver is None:
        try:
            theResolver = createResolver()
        except ValueError:
            theResolver = createResolver(servers=[('127.0.0.1', 53)])
    return theResolver