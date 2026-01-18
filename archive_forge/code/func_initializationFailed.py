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
def initializationFailed(self, reason):
    """
        Called when stream initialization has failed.

        Stream initialization has halted, with the reason indicated by
        C{reason}. It may be retried by calling the authenticator's
        C{initializeStream}. See the respective authenticators for details.

        @param reason: A failure instance indicating why stream initialization
                       failed.
        @type reason: L{failure.Failure}
        """