import array
import struct
from . import errors
from .io import gfile
def crc_update(crc, data):
    """Update CRC-32C checksum with data.

    Args:
      crc: 32-bit checksum to update as long.
      data: byte array, string or iterable over bytes.
    Returns:
      32-bit updated CRC-32C as long.
    """
    if type(data) != array.array or data.itemsize != 1:
        buf = array.array('B', data)
    else:
        buf = data
    crc ^= _MASK
    for b in buf:
        table_index = (crc ^ b) & 255
        crc = (CRC_TABLE[table_index] ^ crc >> 8) & _MASK
    return crc ^ _MASK