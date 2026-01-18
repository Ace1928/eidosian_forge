import array
import struct
from . import errors
from .io import gfile
def masked_crc32c(data):
    x = u32(crc32c(data))
    return u32((x >> 15 | u32(x << 17)) + 2726488792)