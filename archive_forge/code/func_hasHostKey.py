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
def hasHostKey(self, hostname, key):
    """
        Check for an entry with matching hostname and key.

        @param hostname: A hostname or IP address literal to check for.
        @type hostname: L{bytes}

        @param key: The public key to check for.
        @type key: L{Key}

        @return: C{True} if the given hostname and key are present in this file,
            C{False} if they are not.
        @rtype: L{bool}

        @raise HostKeyChanged: if the host key found for the given hostname
            does not match the given key.
        """
    for lineidx, entry in enumerate(self.iterentries(), -len(self._added)):
        if entry.matchesHost(hostname) and entry.keyType == key.sshType():
            if entry.matchesKey(key):
                return True
            else:
                if lineidx < 0:
                    line = None
                    path = None
                else:
                    line = lineidx + 1
                    path = self._savePath
                raise HostKeyChanged(entry, path, line)
    return False