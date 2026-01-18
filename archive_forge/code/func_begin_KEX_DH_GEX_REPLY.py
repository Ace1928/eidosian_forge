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
def begin_KEX_DH_GEX_REPLY(self):
    """
        Utility for test_KEX_DH_GEX_REPLY and
        test_disconnectGEX_REPLYBadSignature.

        Begins a Diffie-Hellman key exchange in an unnamed
        (server-specified) group and computes information needed to
        return either a correct or incorrect signature.
        """
    self.test_KEX_DH_GEX_GROUP()
    p = self.proto.p
    f = 3
    fMP = common.MP(f)
    sharedSecret = _MPpow(f, self.proto.dhSecretKey.private_numbers().x, p)
    h = self.hashProcessor()
    h.update(common.NS(self.proto.ourVersionString) * 2)
    h.update(common.NS(self.proto.ourKexInitPayload) * 2)
    h.update(common.NS(self.blob))
    h.update(b'\x00\x00\x04\x00\x00\x00\x08\x00\x00\x00 \x00')
    h.update(common.MP(self.P1536) + common.MP(2))
    h.update(self.proto.dhSecretKeyPublicMP)
    h.update(fMP)
    h.update(sharedSecret)
    exchangeHash = h.digest()
    signature = self.privObj.sign(exchangeHash)
    return (exchangeHash, signature, common.NS(self.blob) + fMP)