import struct
from twisted.conch import error, interfaces
from twisted.conch.ssh import keys, service, transport
from twisted.conch.ssh.common import NS, getNS
from twisted.cred import credentials
from twisted.cred.error import UnauthorizedLogin
from twisted.internet import defer, reactor
from twisted.logger import Logger
from twisted.python import failure
from twisted.python.compat import nativeString
def ssh_USERAUTH_FAILURE(self, packet):
    """
        We received a MSG_USERAUTH_FAILURE.  Payload::
            string methods
            byte partial success

        If partial success is C{True}, then the previous method succeeded but is
        not sufficient for authentication. C{methods} is a comma-separated list
        of accepted authentication methods.

        We sort the list of methods by their position in C{self.preferredOrder},
        removing methods that have already succeeded. We then call
        C{self.tryAuth} with the most preferred method.

        @param packet: the C{MSG_USERAUTH_FAILURE} payload.
        @type packet: L{bytes}

        @return: a L{defer.Deferred} that will be callbacked with L{None} as
            soon as all authentication methods have been tried, or L{None} if no
            more authentication methods are available.
        @rtype: C{defer.Deferred} or L{None}
        """
    canContinue, partial = getNS(packet)
    partial = ord(partial)
    if partial:
        self.authenticatedWith.append(self.lastAuth)

    def orderByPreference(meth):
        """
            Invoked once per authentication method in order to extract a
            comparison key which is then used for sorting.

            @param meth: the authentication method.
            @type meth: L{bytes}

            @return: the comparison key for C{meth}.
            @rtype: L{int}
            """
        if meth in self.preferredOrder:
            return self.preferredOrder.index(meth)
        else:
            return len(self.preferredOrder)
    canContinue = sorted((meth for meth in canContinue.split(b',') if meth not in self.authenticatedWith), key=orderByPreference)
    self._log.debug('can continue with: {methods}', methods=canContinue)
    return self._cbUserauthFailure(None, iter(canContinue))