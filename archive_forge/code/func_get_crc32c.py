from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import warnings
import six
def get_crc32c(initial_data=b''):
    """Returns an instance of Hashlib-like helper for CRC32C operations.

  Args:
    initial_data (bytes): The CRC32C object will be initialized with the
      checksum of the data.

  Returns:
    The google_crc32c.Checksum instance
    if google-crc32c (https://github.com/googleapis/python-crc32c) is
    available. If not, returns the predefined.Crc instance from crcmod library.

  Usage:
    # Get the instance.
    crc = get_crc32c()
    # Update the instance with data. If your data is available in chunks,
    # you can update each chunk so that you don't have to keep everything in
    # memory.
    for chunk in chunks:
      crc.update(data)
    # Get the digest.
    crc_digest = crc.digest()

  """
    if IS_FAST_GOOGLE_CRC32C_AVAILABLE:
        crc = google_crc32c.Checksum()
    else:
        crc = crcmod.predefined.Crc('crc-32c')
    if initial_data:
        crc.update(initial_data)
    return crc