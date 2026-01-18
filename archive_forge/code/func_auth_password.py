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
def auth_password(self):
    """
        Try to authenticate with a password.  Ask the user for a password.
        If the user will return a password, return True.  Otherwise, return
        False.

        @rtype: L{bool}
        """
    d = self.getPassword()
    if d:
        d.addCallbacks(self._cbPassword, self._ebAuth)
        return True
    else:
        return False