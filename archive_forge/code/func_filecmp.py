import binascii
import os
from posixpath import join as urljoin
import uuid
import six
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import _pywrap_file_io
from tensorflow.python.util import compat
from tensorflow.python.util import deprecation
from tensorflow.python.util.tf_export import tf_export
def filecmp(filename_a, filename_b):
    """Compare two files, returning True if they are the same, False otherwise.

  We check size first and return False quickly if the files are different sizes.
  If they are the same size, we continue to generating a crc for the whole file.

  You might wonder: why not use Python's `filecmp.cmp()` instead? The answer is
  that the builtin library is not robust to the many different filesystems
  TensorFlow runs on, and so we here perform a similar comparison with
  the more robust FileIO.

  Args:
    filename_a: string path to the first file.
    filename_b: string path to the second file.

  Returns:
    True if the files are the same, False otherwise.
  """
    size_a = FileIO(filename_a, 'rb').size()
    size_b = FileIO(filename_b, 'rb').size()
    if size_a != size_b:
        return False
    crc_a = file_crc32(filename_a)
    crc_b = file_crc32(filename_b)
    return crc_a == crc_b