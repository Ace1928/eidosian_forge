from __future__ import print_function
import re
import struct
import binascii
from collections import namedtuple
from Cryptodome.Util.py3compat import bord, tobytes, tostr, bchr, is_string
from Cryptodome.Util.number import bytes_to_long, long_to_bytes
from Cryptodome.Math.Numbers import Integer
from Cryptodome.Util.asn1 import (DerObjectId, DerOctetString, DerSequence,
from Cryptodome.Util._raw_api import (load_pycryptodome_raw_lib, VoidPointer,
from Cryptodome.PublicKey import (_expand_subject_public_key_info,
from Cryptodome.Hash import SHA512, SHAKE256
from Cryptodome.Random import get_random_bytes
from Cryptodome.Random.random import getrandbits
def lib_func(ecc_obj, func_name):
    if ecc_obj._curve.desc == 'Ed25519':
        result = getattr(_ed25519_lib, 'ed25519_' + func_name)
    elif ecc_obj._curve.desc == 'Ed448':
        result = getattr(_ed448_lib, 'ed448_' + func_name)
    else:
        result = getattr(_ec_lib, 'ec_ws_' + func_name)
    return result