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
def init_ed448():
    p = 726838724295606890549323807888004534353641360687318060281490199180612328166730772686396383698676545930088884461843637361053498018365439
    order = 181709681073901722637330951972001133588410340171829515070372549795146003961539585716195755291692375963310293709091662304773755859649779
    Gx = 224580040295924300187604334099896036246789641632564134246125461686950415467406032909029192869357953282578032075146446173674602635247710
    Gy = 298819210078481492676017930443930673437544040154080242095928241372331506189835876003536878655418784733982303233503462500531545062832660
    ed448_context = VoidPointer()
    result = _ed448_lib.ed448_new_context(ed448_context.address_of())
    if result:
        raise ImportError('Error %d initializing Ed448 context' % result)
    context = SmartPointer(ed448_context.get(), _ed448_lib.ed448_free_context)
    ed448 = _Curve(Integer(p), None, Integer(order), Integer(Gx), Integer(Gy), None, 448, '1.3.101.113', context, 'Ed448', None, 'ed448')
    global ed448_names
    _curves.update(dict.fromkeys(ed448_names, ed448))