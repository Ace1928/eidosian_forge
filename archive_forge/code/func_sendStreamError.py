from binascii import hexlify
from hashlib import sha1
from sys import intern
from typing import Optional, Tuple
from zope.interface import directlyProvides, implementer
from twisted.internet import defer, protocol
from twisted.internet.error import ConnectionLost
from twisted.python import failure, log, randbytes
from twisted.words.protocols.jabber import error, ijabber, jid
from twisted.words.xish import domish, xmlstream
from twisted.words.xish.xmlstream import (
def sendStreamError(self, streamError):
    """
        Send stream level error.

        If we are the receiving entity, and haven't sent the header yet,
        we sent one first.

        After sending the stream error, the stream is closed and the transport
        connection dropped.

        @param streamError: stream error instance
        @type streamError: L{error.StreamError}
        """
    if not self._headerSent and (not self.initiating):
        self.sendHeader()
    if self._headerSent:
        self.send(streamError.getElement())
        self.sendFooter()
    self.transport.loseConnection()