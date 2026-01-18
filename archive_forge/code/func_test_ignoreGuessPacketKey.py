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
def test_ignoreGuessPacketKey(self):
    """
        Like test_ignoreGuessPacketKex, but for an incorrectly guessed
        public key format.
        """
    kexInitPacket = b'\x00' * 16 + b''.join([common.NS(x) for x in [b','.join(y) for y in [self.proto.supportedKeyExchanges, self.proto.supportedPublicKeys[::-1], self.proto.supportedCiphers, self.proto.supportedCiphers, self.proto.supportedMACs, self.proto.supportedMACs, self.proto.supportedCompressions, self.proto.supportedCompressions, self.proto.supportedLanguages, self.proto.supportedLanguages]]]) + b'\xff\x00\x00\x00\x00'
    self.proto.ssh_KEXINIT(kexInitPacket)
    self.assertTrue(self.proto.ignoreNextPacket)
    self.proto.ssh_DEBUG(b'\x01\x00\x00\x00\x04test\x00\x00\x00\x00')
    self.assertTrue(self.proto.ignoreNextPacket)
    self.proto.ssh_KEX_DH_GEX_REQUEST_OLD(b'\x00\x00\x08\x00')
    self.assertFalse(self.proto.ignoreNextPacket)
    self.assertEqual(self.packets, [])
    self.proto.ignoreNextPacket = True
    self.proto.ssh_KEX_DH_GEX_REQUEST(b'\x00\x00\x08\x00' * 3)
    self.assertFalse(self.proto.ignoreNextPacket)
    self.assertEqual(self.packets, [])