from .lib.py3compat import int2byte
from .lib import (BitStreamReader, BitStreamWriter, encode_bin,
from .core import (Struct, MetaField, StaticField, FormatField,
from .adapters import (BitIntegerAdapter, PaddingAdapter,
def resizer(length):
    if length & 7:
        raise SizeofError('size must be a multiple of 8', length)
    return length >> 3