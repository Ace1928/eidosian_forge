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
def test__getHostKeys(self):
    """
        L{transport.SSHServerTransport._getHostKeys} returns host keys from
        the factory, looked up by public key signature algorithm.
        """
    self.proto.factory.publicKeys = {b'ssh-rsa': keys.Key.fromString(keydata.publicRSA_openssh), b'ssh-dss': keys.Key.fromString(keydata.publicDSA_openssh), b'ecdsa-sha2-nistp256': keys.Key.fromString(keydata.publicECDSA_openssh), b'ssh-ed25519': keys.Key.fromString(keydata.publicEd25519_openssh)}
    self.proto.factory.privateKeys = {b'ssh-rsa': keys.Key.fromString(keydata.privateRSA_openssh), b'ssh-dss': keys.Key.fromString(keydata.privateDSA_openssh), b'ecdsa-sha2-nistp256': keys.Key.fromString(keydata.privateECDSA_openssh), b'ssh-ed25519': keys.Key.fromString(keydata.privateEd25519_openssh_new)}
    self.assertEqual(self.proto._getHostKeys(b'ssh-rsa'), (self.proto.factory.publicKeys[b'ssh-rsa'], self.proto.factory.privateKeys[b'ssh-rsa']))
    self.assertEqual(self.proto._getHostKeys(b'rsa-sha2-256'), (self.proto.factory.publicKeys[b'ssh-rsa'], self.proto.factory.privateKeys[b'ssh-rsa']))
    self.assertEqual(self.proto._getHostKeys(b'rsa-sha2-512'), (self.proto.factory.publicKeys[b'ssh-rsa'], self.proto.factory.privateKeys[b'ssh-rsa']))
    self.assertEqual(self.proto._getHostKeys(b'ssh-dss'), (self.proto.factory.publicKeys[b'ssh-dss'], self.proto.factory.privateKeys[b'ssh-dss']))
    self.assertEqual(self.proto._getHostKeys(b'ecdsa-sha2-nistp256'), (self.proto.factory.publicKeys[b'ecdsa-sha2-nistp256'], self.proto.factory.privateKeys[b'ecdsa-sha2-nistp256']))
    self.assertEqual(self.proto._getHostKeys(b'ssh-ed25519'), (self.proto.factory.publicKeys[b'ssh-ed25519'], self.proto.factory.privateKeys[b'ssh-ed25519']))
    self.assertRaises(KeyError, self.proto._getHostKeys, b'ecdsa-sha2-nistp384')