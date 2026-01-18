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
def ssh_USERAUTH_PK_OK_publickey(self, packet):
    """
        This is MSG_USERAUTH_PK.  Our public key is valid, so we create a
        signature and try to authenticate with it.
        """
    publicKey = self.lastPublicKey
    b = NS(self.transport.sessionID) + bytes((MSG_USERAUTH_REQUEST,)) + NS(self.user) + NS(self.instance.name) + NS(b'publickey') + b'\x01' + NS(publicKey.sshType()) + NS(publicKey.blob())
    d = self.signData(publicKey, b)
    if not d:
        self.askForAuth(b'none', b'')
        return
    d.addCallback(self._cbSignedData)
    d.addErrback(self._ebAuth)