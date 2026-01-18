from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def get_checksum(crc):
    """Gets the hex checksum from a CRC32C object.

  Args:
    crc (google_crc32c.Checksum|predefined.Crc): CRC32C object from
      google-crc32c or crcmod package.

  Returns:
    An int representing the CRC32C checksum of the provided object.
  """
    return int(crc.hexdigest(), 16)