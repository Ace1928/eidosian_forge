from __future__ import annotations
import binascii
import struct
import unicodedata
import warnings
from base64 import b64encode, decodebytes, encodebytes
from hashlib import md5, sha256
from typing import Any
import bcrypt
from cryptography import utils
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import dsa, ec, ed25519, padding, rsa
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.serialization import (
from typing_extensions import Literal
from twisted.conch.ssh import common, sexpy
from twisted.conch.ssh.common import int_to_bytes
from twisted.python import randbytes
from twisted.python.compat import iterbytes, nativeString
from twisted.python.constants import NamedConstant, Names
from twisted.python.deprecate import _mutuallyExclusiveArguments
def sshType(self):
    """
        Get the type of the object we wrap as defined in the SSH protocol,
        defined in RFC 4253, Section 6.6 and RFC 8332, section 4 (this is a
        public key format name, not a public key algorithm name). Currently
        this can only be b'ssh-rsa', b'ssh-dss', b'ecdsa-sha2-[identifier]'
        or b'ssh-ed25519'.

        identifier is the standard NIST curve name

        @return: The key type format.
        @rtype: L{bytes}
        """
    if self.type() == 'EC':
        return b'ecdsa-sha2-' + _secToNist[self._keyObject.curve.name.encode('ascii')]
    else:
        return {'RSA': b'ssh-rsa', 'DSA': b'ssh-dss', 'Ed25519': b'ssh-ed25519'}[self.type()]