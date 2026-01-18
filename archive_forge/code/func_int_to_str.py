import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def int_to_str(int_val, dialect=None):
    """
    :param int_val: An unsigned integer.

    :param dialect: (optional) a Python class defining formatting options

    :return: An IEEE EUI-64 identifier that is equivalent to unsigned integer.
    """
    if dialect is None:
        dialect = eui64_base
    words = int_to_words(int_val, dialect)
    tokens = [dialect.word_fmt % i for i in words]
    addr = dialect.word_sep.join(tokens)
    return addr