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
@classmethod
def fromPath(cls, path):
    """
        Create a new L{KnownHostsFile}, potentially reading existing known
        hosts information from the given file.

        @param path: A path object to use for both reading contents from and
            later saving to.  If no file exists at this path, it is not an
            error; a L{KnownHostsFile} with no entries is returned.
        @type path: L{FilePath}

        @return: A L{KnownHostsFile} initialized with entries from C{path}.
        @rtype: L{KnownHostsFile}
        """
    knownHosts = cls(path)
    knownHosts._clobber = False
    return knownHosts