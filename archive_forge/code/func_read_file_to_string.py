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
def read_file_to_string(filename, binary_mode=False):
    """Reads the entire contents of a file to a string.

  Args:
    filename: string, path to a file
    binary_mode: whether to open the file in binary mode or not. This changes
      the type of the object returned.

  Returns:
    contents of the file as a string or bytes.

  Raises:
    errors.OpError: Raises variety of errors that are subtypes e.g.
    `NotFoundError` etc.
  """
    if binary_mode:
        f = FileIO(filename, mode='rb')
    else:
        f = FileIO(filename, mode='r')
    return f.read()