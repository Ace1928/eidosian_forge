import binascii
import struct
from Cryptodome import Random
from Cryptodome.Util.py3compat import tobytes, bord, tostr
from Cryptodome.Util.asn1 import DerSequence, DerNull
from Cryptodome.Util.number import bytes_to_long
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Math.Primality import (test_probable_prime,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
importKey = import_key
@property
def invq(self):
    if not self.has_private():
        raise AttributeError("No CRT component 'invq' available for public keys")
    if self._invq is None:
        self._invq = self._q.inverse(self._p)
    return int(self._invq)