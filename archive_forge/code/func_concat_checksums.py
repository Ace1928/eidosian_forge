from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def concat_checksums(crc_a, crc_b, b_byte_count):
    """Computes CRC32C for concat(A, B) given crc(A), crc(B),and len(B).

  An explanation of the algorithm can be found at
  https://code.google.com/archive/p/crcutil/downloads.

  Args:
    crc_a (int): Represents the CRC32C checksum of object A.
    crc_b (int): Represents the CRC32C checksum of object B.
    b_byte_count (int): Length of data covered by crc_b in bytes.

  Returns:
    CRC32C checksum representing the data covered by crc_a and crc_b (int).
  """
    if not b_byte_count:
        return crc_a
    b_bit_count = 8 * b_byte_count
    return _extend_crc32c_checksum_by_zeros(crc_a, bit_count=b_bit_count) ^ crc_b