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
def begin_KEXDH_REPLY(self):
    """
        Utility for test_KEXDH_REPLY and
        test_disconnectKEXDH_REPLYBadSignature.

        Begins an Elliptic Curve Diffie-Hellman key exchange and computes
        information needed to return either a correct or incorrect
        signature.
        """
    self.test_KEXINIT()
    privKey = MockFactory().getPrivateKeys()[b'ssh-rsa']
    pubKey = MockFactory().getPublicKeys()[b'ssh-rsa']
    ecPriv = self.proto._generateECPrivateKey()
    ecPub = ecPriv.public_key()
    encPub = self.proto._encodeECPublicKey(ecPub)
    sharedSecret = self.proto._generateECSharedSecret(ecPriv, self.proto._encodeECPublicKey(self.proto.ecPub))
    h = self.hashProcessor()
    h.update(common.NS(self.proto.ourVersionString))
    h.update(common.NS(self.proto.otherVersionString))
    h.update(common.NS(self.proto.ourKexInitPayload))
    h.update(common.NS(self.proto.otherKexInitPayload))
    h.update(common.NS(pubKey.blob()))
    h.update(common.NS(self.proto._encodeECPublicKey(self.proto.ecPub)))
    h.update(common.NS(encPub))
    h.update(sharedSecret)
    exchangeHash = h.digest()
    signature = privKey.sign(exchangeHash)
    return (exchangeHash, signature, common.NS(pubKey.blob()) + common.NS(encPub))