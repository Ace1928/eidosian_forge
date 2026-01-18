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
def auth_keyboard_interactive(self):
    """
        Try to authenticate with keyboard-interactive authentication.  Send
        the request to the server and return True.

        @rtype: L{bool}
        """
    self._log.debug('authing with keyboard-interactive')
    self.askForAuth(b'keyboard-interactive', NS(b'') + NS(b''))
    return True