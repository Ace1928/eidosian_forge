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
def init_p224():
    p = 26959946667150639794667015087019630673557916260026308143510066298881
    b = 18958286285566608000408668544493926415504680968679321075787234672564
    order = 26959946667150639794667015087019625940457807714424391721682722368061
    Gx = 19277929113566293071110308034699488026831934219452440156649784352033
    Gy = 19926808758034470970197974370888749184205991990603949537637343198772
    p224_modulus = long_to_bytes(p, 28)
    p224_b = long_to_bytes(b, 28)
    p224_order = long_to_bytes(order, 28)
    ec_p224_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p224_context.address_of(), c_uint8_ptr(p224_modulus), c_uint8_ptr(p224_b), c_uint8_ptr(p224_order), c_size_t(len(p224_modulus)), c_ulonglong(getrandbits(64)))
    if result:
        raise ImportError('Error %d initializing P-224 context' % result)
    context = SmartPointer(ec_p224_context.get(), _ec_lib.ec_free_context)
    p224 = _Curve(Integer(p), Integer(b), Integer(order), Integer(Gx), Integer(Gy), None, 224, '1.3.132.0.33', context, 'NIST P-224', 'ecdsa-sha2-nistp224', 'p224')
    global p224_names
    _curves.update(dict.fromkeys(p224_names, p224))