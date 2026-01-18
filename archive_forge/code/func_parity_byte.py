import sys
from Cryptodome.Cipher import _create_cipher
from Cryptodome.Util.py3compat import byte_string, bchr, bord, bstr
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib,
def parity_byte(key_byte):
    parity = 1
    for i in range(1, 8):
        parity ^= key_byte >> i & 1
    return key_byte & 254 | parity