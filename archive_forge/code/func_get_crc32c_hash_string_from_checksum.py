from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def get_crc32c_hash_string_from_checksum(checksum):
    """Returns base64-encoded hash from the checksum.

  Args:
    checksum (int): CRC32C checksum representing the hash of processed data.

  Returns:
    A string representing the base64 encoded CRC32C hash.
  """
    crc_object = get_crc32c_from_checksum(checksum)
    return get_hash(crc_object)