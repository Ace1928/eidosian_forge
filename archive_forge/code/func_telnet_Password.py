import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def telnet_Password(self, line):
    username, password = (self.username, line)
    del self.username

    def login(ignored):
        creds = credentials.UsernamePassword(username, password)
        d = self.portal.login(creds, None, ITelnetProtocol)
        d.addCallback(self._cbLogin)
        d.addErrback(self._ebLogin)
    self.transport.wont(ECHO).addCallback(login)
    return 'Discard'