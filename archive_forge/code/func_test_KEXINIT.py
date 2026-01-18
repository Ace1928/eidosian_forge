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
def test_KEXINIT(self):
    """
        KEXINIT packet with an elliptic curve key exchange results
        in a KEXDH_INIT message.
        """
    self.proto.supportedKeyExchanges = [self.kexAlgorithm]
    self.proto.dataReceived(self.transport.value())
    self.assertEqual(self.packets, [(transport.MSG_KEXDH_INIT, common.NS(self.proto._encodeECPublicKey(self.proto.ecPub)))])