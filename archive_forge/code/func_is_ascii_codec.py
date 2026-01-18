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
def is_ascii_codec(codec):
    """Test if codec is compatible with 7-bit ascii (e.g. latin-1, utf-8; but not utf-16)"""
    return _ASCII_TEST_UNICODE.encode(codec) == _ASCII_TEST_BYTES