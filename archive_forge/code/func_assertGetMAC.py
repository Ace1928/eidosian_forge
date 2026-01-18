import binascii
import re
import string
import struct
import types
from hashlib import md5, sha1, sha256, sha384, sha512
from typing import Dict, List, Optional, Tuple, Type
from twisted import __version__ as twisted_version
from twisted.conch.error import ConchError
from twisted.conch.ssh import _kex, address, service
from twisted.internet import defer
from twisted.protocols import loopback
from twisted.python import randbytes
from twisted.python.compat import iterbytes
from twisted.python.randbytes import insecureRandom
from twisted.python.reflect import requireModule
from twisted.test import proto_helpers
from twisted.trial.unittest import TestCase
def assertGetMAC(self, hmacName, hashProcessor, digestSize, blockPadSize):
    """
        Check that when L{SSHCiphers._getMAC} is called with a supportd HMAC
        algorithm name it returns a tuple of
        (digest object, inner pad, outer pad, digest size) with a C{key}
        attribute set to the value of the key supplied.

        @param hmacName: Identifier of HMAC algorithm.
        @type hmacName: L{bytes}

        @param hashProcessor: Callable for the hash algorithm.
        @type hashProcessor: C{callable}

        @param digestSize: Size of the digest for algorithm.
        @type digestSize: L{int}

        @param blockPadSize: Size of padding applied to the shared secret to
            match the block size.
        @type blockPadSize: L{int}
        """
    secret = self.getSharedSecret()
    params = self.ciphers._getMAC(hmacName, secret)
    key = secret[:digestSize] + b'\x00' * blockPadSize
    innerPad = bytes((ord(b) ^ 54 for b in iterbytes(key)))
    outerPad = bytes((ord(b) ^ 92 for b in iterbytes(key)))
    self.assertEqual((hashProcessor, innerPad, outerPad, digestSize), params)
    self.assertEqual(key, params.key)