from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import hashlib
def get_md5(byte_string=b''):
    """Returns md5 object, avoiding incorrect FIPS error on Red Hat systems.

  Examples: get_md5(b'abc')
            get_md5(bytes('abc', encoding='utf-8'))

  Args:
    byte_string (bytes): String in bytes form to hash. Don't include for empty
      hash object, since md5(b'').digest() == md5().digest().

  Returns:
    md5 hash object.
  """
    try:
        return hashlib.md5(byte_string)
    except ValueError:
        return hashlib.md5(byte_string, usedforsecurity=False)