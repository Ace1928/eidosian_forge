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
def assertKexDHInitResponse(self, kexAlgorithm, keyAlgorithm, bits):
    """
        Test that the KEXDH_INIT packet causes the server to send a
        KEXDH_REPLY with the server's public key and a signature.

        @param kexAlgorithm: The key exchange algorithm to use.
        @type kexAlgorithm: L{bytes}

        @param keyAlgorithm: The public key signature algorithm to use.
        @type keyAlgorithm: L{bytes}

        @param bits: The bit length of the DH modulus.
        @type bits: L{int}
        """
    self.proto.supportedKeyExchanges = [kexAlgorithm]
    self.proto.supportedPublicKeys = [keyAlgorithm]
    self.proto.dataReceived(self.transport.value())
    pubHostKey, privHostKey = self.proto._getHostKeys(keyAlgorithm)
    g, p = _kex.getDHGeneratorAndPrime(kexAlgorithm)
    e = pow(g, 5000, p)
    self.proto.ssh_KEX_DH_GEX_REQUEST_OLD(common.MP(e))
    y = common.getMP(common.NS(b'\x99' * (bits // 8)))[0]
    f = _MPpow(self.proto.g, y, self.proto.p)
    self.assertEqual(self.proto.dhSecretKeyPublicMP, f)
    sharedSecret = _MPpow(e, y, self.proto.p)
    h = sha1()
    h.update(common.NS(self.proto.ourVersionString) * 2)
    h.update(common.NS(self.proto.ourKexInitPayload) * 2)
    h.update(common.NS(pubHostKey.blob()))
    h.update(common.MP(e))
    h.update(f)
    h.update(sharedSecret)
    exchangeHash = h.digest()
    signature = privHostKey.sign(exchangeHash, signatureType=keyAlgorithm)
    self.assertEqual(self.packets, [(transport.MSG_KEXDH_REPLY, common.NS(pubHostKey.blob()) + f + common.NS(signature)), (transport.MSG_NEWKEYS, b'')])