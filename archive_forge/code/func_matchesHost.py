import hmac
import sys
from binascii import Error as DecodeError, a2b_base64, b2a_base64
from contextlib import closing
from hashlib import sha1
from zope.interface import implementer
from twisted.conch.error import HostKeyChanged, InvalidEntry, UserRejectedKey
from twisted.conch.interfaces import IKnownHostEntry
from twisted.conch.ssh.keys import BadKeyError, FingerprintFormats, Key
from twisted.internet import defer
from twisted.logger import Logger
from twisted.python.compat import nativeString
from twisted.python.randbytes import secureRandom
from twisted.python.util import FancyEqMixin
def matchesHost(self, hostname):
    """
        Implement L{IKnownHostEntry.matchesHost} to compare the hash of the
        input to the stored hash.

        @param hostname: A hostname or IP address literal to check against this
            entry.
        @type hostname: L{bytes}

        @return: C{True} if this entry is for the given hostname or IP address,
            C{False} otherwise.
        @rtype: L{bool}
        """
    return hmac.compare_digest(_hmacedString(self._hostSalt, hostname), self._hostHash)