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
def filterAnswers(self, message):
    """
        Extract results from the given message.

        If the message was truncated, re-attempt the query over TCP and return
        a Deferred which will fire with the results of that query.

        If the message's result code is not C{twisted.names.dns.OK}, return a
        Failure indicating the type of error which occurred.

        Otherwise, return a three-tuple of lists containing the results from
        the answers section, the authority section, and the additional section.
        """
    if message.trunc:
        return self.queryTCP(message.queries).addCallback(self.filterAnswers)
    if message.rCode != dns.OK:
        return failure.Failure(self.exceptionForCode(message.rCode)(message))
    return (message.answers, message.authority, message.additional)