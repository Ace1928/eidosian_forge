import struct
from zope.interface import implementer
from twisted.internet import defer, interfaces as iinternet, protocol
from twisted.logger import Logger
from twisted.python.compat import iterbytes
from twisted.protocols import basic
from twisted.cred import credentials
def requestNegotiation(self, about, data):
    """
        Send a negotiation message for the option C{about} with C{data} as the
        payload.

        @param data: the payload
        @type data: L{bytes}
        @see: L{ITelnetTransport.requestNegotiation}
        """
    data = data.replace(IAC, IAC * 2)
    self._write(IAC + SB + about + data + IAC + SE)