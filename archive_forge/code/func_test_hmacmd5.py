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
def test_hmacmd5(self):
    """
        When L{SSHCiphers._getMAC} is called with the C{b"hmac-md5"} MAC
        algorithm name it returns a tuple of (md5 digest object, inner pad,
        outer pad, md5 digest size) with a C{key} attribute set to the value of
        the key supplied.
        """
    self.assertGetMAC(b'hmac-md5', md5, digestSize=16, blockPadSize=48)