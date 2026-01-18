import struct
from binascii import unhexlify
from Cryptodome.Util.py3compat import bord, _copy_bytes, bchr
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
from Cryptodome.Util.strxor import strxor
from Cryptodome.Hash import BLAKE2s
from Cryptodome.Random import get_random_bytes
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
def encrypt_and_digest(self, plaintext):
    """Encrypt the message and create the MAC tag in one step.

        :Parameters:
          plaintext : bytes/bytearray/memoryview
            The entire message to encrypt.
        :Return:
            a tuple with two byte strings:

            - the encrypted data
            - the MAC
        """
    return (self.encrypt(plaintext) + self.encrypt(), self.digest())