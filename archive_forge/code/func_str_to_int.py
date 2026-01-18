import struct as _struct
import re as _re
from netaddr.core import AddrFormatError
from netaddr.strategy import (
def str_to_int(addr):
    """
    :param addr: An IEEE EUI-64 identifier in string form.

    :return: An unsigned integer that is equivalent to value represented
        by EUI-64 string address formatted according to the dialect
    """
    words = []
    try:
        words = _get_match_result(addr, RE_EUI64_FORMATS)
        if not words:
            raise TypeError
    except TypeError:
        raise AddrFormatError('invalid IEEE EUI-64 identifier: %r!' % (addr,))
    if isinstance(words, tuple):
        pass
    else:
        words = (words,)
    if len(words) == 8:
        int_val = int(''.join(['%.2x' % int(w, 16) for w in words]), 16)
    elif len(words) == 4:
        int_val = int(''.join(['%.4x' % int(w, 16) for w in words]), 16)
    elif len(words) == 1:
        int_val = int('%016x' % int(words[0], 16), 16)
    else:
        raise AddrFormatError('bad word count for EUI-64 identifier: %r!' % addr)
    return int_val