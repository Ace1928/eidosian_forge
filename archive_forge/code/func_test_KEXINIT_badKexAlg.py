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
def test_KEXINIT_badKexAlg(self):
    """
        Test that the client raises a ConchError if it receives a
        KEXINIT message but doesn't have a key exchange algorithm that we
        understand.
        """
    self.proto.supportedKeyExchanges = [b'diffie-hellman-group24-sha1']
    data = self.transport.value().replace(b'group14', b'group24')
    self.assertRaises(ConchError, self.proto.dataReceived, data)