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
def test_KEXINITMultipleAlgorithms(self):
    """
        Receiving a KEXINIT packet listing multiple supported
        algorithms will set up the first common algorithm, ordered after our
        preference.
        """
    self.proto.dataReceived(b'SSH-2.0-Twisted\r\n\x00\x00\x01\xf4\x04\x14\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x99\x00\x00\x00bdiffie-hellman-group1-sha1,diffie-hellman-group-exchange-sha1,diffie-hellman-group-exchange-sha256\x00\x00\x00\x0fssh-dss,ssh-rsa\x00\x00\x00\x85aes128-ctr,aes128-cbc,aes192-ctr,aes192-cbc,aes256-ctr,aes256-cbc,cast128-ctr,cast128-cbc,blowfish-ctr,blowfish-cbc,3des-ctr,3des-cbc\x00\x00\x00\x85aes128-ctr,aes128-cbc,aes192-ctr,aes192-cbc,aes256-ctr,aes256-cbc,cast128-ctr,cast128-cbc,blowfish-ctr,blowfish-cbc,3des-ctr,3des-cbc\x00\x00\x00\x12hmac-md5,hmac-sha1\x00\x00\x00\x12hmac-md5,hmac-sha1\x00\x00\x00\tzlib,none\x00\x00\x00\tzlib,none\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x99\x99\x99\x99')
    self.assertEqual(self.proto.kexAlg, b'diffie-hellman-group-exchange-sha256')
    self.assertEqual(self.proto.keyAlg, b'ssh-rsa')
    self.assertEqual(self.proto.outgoingCompressionType, b'none')
    self.assertEqual(self.proto.incomingCompressionType, b'none')
    ne = self.proto.nextEncryptions
    self.assertEqual(ne.outCipType, b'aes256-ctr')
    self.assertEqual(ne.inCipType, b'aes256-ctr')
    self.assertEqual(ne.outMACType, b'hmac-sha1')
    self.assertEqual(ne.inMACType, b'hmac-sha1')