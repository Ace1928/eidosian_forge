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
def askForAuth(self, kind, extraData):
    """
        Send a MSG_USERAUTH_REQUEST.

        @param kind: the authentication method to try.
        @type kind: L{bytes}
        @param extraData: method-specific data to go in the packet
        @type extraData: L{bytes}
        """
    self.lastAuth = kind
    self.transport.sendPacket(MSG_USERAUTH_REQUEST, NS(self.user) + NS(self.instance.name) + NS(kind) + extraData)