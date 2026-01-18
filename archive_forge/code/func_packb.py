import functools
import catalogue
from ._version import version
from .exceptions import *
from ._packer import Packer as _Packer
from ._unpacker import unpackb as _unpackb
from ._unpacker import unpack as _unpack
from ._unpacker import Unpacker as _Unpacker
from ._ext_type import ExtType
from ._msgpack_numpy import encode_numpy as _encode_numpy
from ._msgpack_numpy import decode_numpy as _decode_numpy
def packb(o, **kwargs):
    """
    Pack an object and return the packed bytes.
    """
    return Packer(**kwargs).pack(o)