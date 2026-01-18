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
def ssh_USERAUTH_REQUEST(self, packet):
    """
        The client has requested authentication.  Payload::
            string user
            string next service
            string method
            <authentication specific data>

        @type packet: L{bytes}
        """
    user, nextService, method, rest = getNS(packet, 3)
    if user != self.user or nextService != self.nextService:
        self.authenticatedWith = []
    self.user = user
    self.nextService = nextService
    self.method = method
    d = self.tryAuth(method, user, rest)
    if not d:
        self._ebBadAuth(failure.Failure(error.ConchError('auth returned none')))
        return
    d.addCallback(self._cbFinishedAuth)
    d.addErrback(self._ebMaybeBadAuth)
    d.addErrback(self._ebBadAuth)
    return d