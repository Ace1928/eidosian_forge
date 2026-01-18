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
def ssh_USERAUTH_PK_OK(self, packet):
    """
        This message (number 60) can mean several different messages depending
        on the current authentication type.  We dispatch to individual methods
        in order to handle this request.
        """
    func = getattr(self, 'ssh_USERAUTH_PK_OK_%s' % nativeString(self.lastAuth.replace(b'-', b'_')), None)
    if func is not None:
        return func(packet)
    else:
        self.askForAuth(b'none', b'')