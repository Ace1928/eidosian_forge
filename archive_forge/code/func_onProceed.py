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
def onProceed(self, obj):
    """
        Proceed with TLS negotiation and reset the XML stream.
        """
    self.xmlstream.removeObserver('/failure', self.onFailure)
    if self._configurationForTLS:
        ctx = self._configurationForTLS
    else:
        ctx = ssl.optionsForClientTLS(self.xmlstream.otherEntity.host)
    self.xmlstream.transport.startTLS(ctx)
    self.xmlstream.reset()
    self.xmlstream.sendHeader()
    self._deferred.callback(Reset)