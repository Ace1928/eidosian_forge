import array
import struct
from . import errors
from .io import gfile
def crc32c(data):
    """Compute CRC-32C checksum of the data.

    Args:
      data: byte array, string or iterable over bytes.
    Returns:
      32-bit CRC-32C checksum of data as long.
    """
    return crc_finalize(crc_update(CRC_INIT, data))