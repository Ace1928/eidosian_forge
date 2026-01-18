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
def getrandstr(rng, charset, count):
    """return string containing *count* number of chars/bytes, whose elements are drawn from specified charset, using specified rng"""
    if count < 0:
        raise ValueError('count must be >= 0')
    letters = len(charset)
    if letters == 0:
        raise ValueError('alphabet must not be empty')
    if letters == 1:
        return charset * count

    def helper():
        value = rng.randrange(0, letters ** count)
        i = 0
        while i < count:
            yield charset[value % letters]
            value //= letters
            i += 1
    if isinstance(charset, unicode):
        return join_unicode(helper())
    else:
        return join_byte_elems(helper())