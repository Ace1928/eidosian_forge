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
def render_bytes(source, *args):
    """Peform ``%`` formating using bytes in a uniform manner across Python 2/3.

    This function is motivated by the fact that
    :class:`bytes` instances do not support ``%`` or ``{}`` formatting under Python 3.
    This function is an attempt to provide a replacement:
    it converts everything to unicode (decoding bytes instances as ``latin-1``),
    performs the required formatting, then encodes the result to ``latin-1``.

    Calling ``render_bytes(source, *args)`` should function roughly the same as
    ``source % args`` under Python 2.

    .. todo::
        python >= 3.5 added back limited support for bytes %,
        can revisit when 3.3/3.4 is dropped.
    """
    if isinstance(source, bytes):
        source = source.decode('latin-1')
    result = source % tuple((arg.decode('latin-1') if isinstance(arg, bytes) else arg for arg in args))
    return result.encode('latin-1')