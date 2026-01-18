import os
import re
import socket
import warnings
from typing import Optional, Sequence, Type
from unicodedata import normalize
from zope.interface import directlyProvides, implementer, provider
from constantly import NamedConstant, Names
from incremental import Version
from twisted.internet import defer, error, fdesc, interfaces, threads
from twisted.internet.abstract import isIPAddress, isIPv6Address
from twisted.internet.address import (
from twisted.internet.interfaces import (
from twisted.internet.protocol import ClientFactory, Factory, ProcessProtocol, Protocol
from twisted.internet._resolver import HostResolution
from twisted.internet.defer import Deferred
from twisted.internet.task import LoopingCall
from twisted.logger import Logger
from twisted.plugin import IPlugin, getPlugins
from twisted.python import deprecate, log
from twisted.python.compat import _matchingString, iterbytes, nativeString
from twisted.python.components import proxyForInterface
from twisted.python.failure import Failure
from twisted.python.filepath import FilePath
from twisted.python.systemd import ListenFDs
from ._idna import _idnaBytes, _idnaText
@d.addCallback
def startConnectionAttempts(endpoints):
    """
            Given a sequence of endpoints obtained via name resolution, start
            connecting to a new one every C{self._attemptDelay} seconds until
            one of the connections succeeds, all of them fail, or the attempt
            is cancelled.

            @param endpoints: a list of all the endpoints we might try to
                connect to, as determined by name resolution.
            @type endpoints: L{list} of L{IStreamServerEndpoint}

            @return: a Deferred that fires with the result of the
                C{endpoint.connect} method that completes the fastest, or fails
                with the first connection error it encountered if none of them
                succeed.
            @rtype: L{Deferred} failing with L{error.ConnectingCancelledError}
                or firing with L{IProtocol}
            """
    if not endpoints:
        raise error.DNSLookupError(f'no results for hostname lookup: {self._hostStr}')
    iterEndpoints = iter(endpoints)
    pending = []
    failures = []
    winner = defer.Deferred(canceller=_canceller)

    def checkDone():
        if pending or checkDone.completed or checkDone.endpointsLeft:
            return
        winner.errback(failures.pop())
    checkDone.completed = False
    checkDone.endpointsLeft = True

    @LoopingCall
    def iterateEndpoint():
        endpoint = next(iterEndpoints, None)
        if endpoint is None:
            checkDone.endpointsLeft = False
            checkDone()
            return
        eachAttempt = endpoint.connect(protocolFactory)
        pending.append(eachAttempt)

        @eachAttempt.addBoth
        def noLongerPending(result):
            pending.remove(eachAttempt)
            return result

        @eachAttempt.addCallback
        def succeeded(result):
            winner.callback(result)

        @eachAttempt.addErrback
        def failed(reason):
            failures.append(reason)
            checkDone()
    iterateEndpoint.clock = self._reactor
    iterateEndpoint.start(self._attemptDelay)

    @winner.addBoth
    def cancelRemainingPending(result):
        checkDone.completed = True
        for remaining in pending[:]:
            remaining.cancel()
        if iterateEndpoint.running:
            iterateEndpoint.stop()
        return result
    return winner