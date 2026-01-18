from __future__ import absolute_import, division, print_function
from passlib.utils.compat import PY3
import base64
import calendar
import json
import logging; log = logging.getLogger(__name__)
import math
import struct
import sys
import time as _time
import re
from warnings import warn
from passlib import exc
from passlib.exc import TokenError, MalformedTokenError, InvalidTokenError, UsedTokenError
from passlib.utils import (to_unicode, to_bytes, consteq,
from passlib.utils.binary import BASE64_CHARS, b32encode, b32decode
from passlib.utils.compat import (u, unicode, native_string_types, bascii_to_str, int_types, num_types,
from passlib.utils.decor import hybrid_method, memoized_property
from passlib.crypto.digest import lookup_hash, compile_hmac, pbkdf2_hmac
from passlib.hash import pbkdf2_sha256
def generate_secret(entropy=256, charset=BASE64_CHARS[:-2]):
    """
    generate a random string suitable for use as an
    :class:`AppWallet` application secret.

    :param entropy:
        number of bits of entropy (controls size/complexity of password).
    """
    assert entropy > 0
    assert len(charset) > 1
    count = int(math.ceil(entropy * math.log(2, len(charset))))
    return getrandstr(rng, charset, count)