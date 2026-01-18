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
def init_p521():
    p = 6864797660130609714981900799081393217269435300143305409394463459185543183397656052122559640661454554977296311391480858037121987999716643812574028291115057151
    b = 1093849038073734274511112390766805569936207598951683748994586394495953116150735016013708737573759623248592132296706313309438452531591012912142327488478985984
    order = 6864797660130609714981900799081393217269435300143305409394463459185543183397655394245057746333217197532963996371363321113864768612440380340372808892707005449
    Gx = 2661740802050217063228768716723360960729859168756973147706671368418802944996427808491545080627771902352094241225065558662157113545570916814161637315895999846
    Gy = 3757180025770020463545507224491183603594455134769762486694567779615544477440556316691234405012945539562144444537289428522585666729196580810124344277578376784
    p521_modulus = long_to_bytes(p, 66)
    p521_b = long_to_bytes(b, 66)
    p521_order = long_to_bytes(order, 66)
    ec_p521_context = VoidPointer()
    result = _ec_lib.ec_ws_new_context(ec_p521_context.address_of(), c_uint8_ptr(p521_modulus), c_uint8_ptr(p521_b), c_uint8_ptr(p521_order), c_size_t(len(p521_modulus)), c_ulonglong(getrandbits(64)))
    if result:
        raise ImportError('Error %d initializing P-521 context' % result)
    context = SmartPointer(ec_p521_context.get(), _ec_lib.ec_free_context)
    p521 = _Curve(Integer(p), Integer(b), Integer(order), Integer(Gx), Integer(Gy), None, 521, '1.3.132.0.35', context, 'NIST P-521', 'ecdsa-sha2-nistp521', 'p521')
    global p521_names
    _curves.update(dict.fromkeys(p521_names, p521))