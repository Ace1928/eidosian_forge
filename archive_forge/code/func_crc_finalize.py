import array
import struct
from . import errors
from .io import gfile
def crc_finalize(crc):
    """Finalize CRC-32C checksum.

    This function should be called as last step of crc calculation.
    Args:
      crc: 32-bit checksum as long.
    Returns:
      finalized 32-bit checksum as long
    """
    return crc & _MASK