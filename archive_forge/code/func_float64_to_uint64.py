import collections
import struct
from . import packer
from .compat import import_numpy, NumpyRequiredForThisFeature
def float64_to_uint64(n):
    packed = struct.pack('<1d', n)
    converted, = struct.unpack('<1Q', packed)
    return converted