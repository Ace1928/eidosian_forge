from passlib.utils.compat import JYTHON
from binascii import b2a_base64, a2b_base64, Error as _BinAsciiError
from base64 import b64encode, b64decode
from codecs import lookup as _lookup_codec
from functools import update_wrapper
import itertools
import inspect
import logging; log = logging.getLogger(__name__)
import math
import os
import sys
import random
import re
import time
import timeit
import types
from warnings import warn
from passlib.utils.binary import (
from passlib.utils.decor import (
from passlib.exc import ExpectedStringError, ExpectedTypeError
from passlib.utils.compat import (add_doc, join_bytes, join_byte_values,
from passlib.exc import MissingBackendError
def safe_crypt(secret, hash):
    if isinstance(secret, unicode):
        secret = secret.encode('utf-8')
    if _NULL in secret:
        raise ValueError('null character in secret')
    if isinstance(hash, unicode):
        hash = hash.encode('ascii')
    with _safe_crypt_lock:
        result = _crypt(secret, hash)
    if not result:
        return None
    result = result.decode('ascii')
    if result[0] in _invalid_prefixes:
        return None
    return result