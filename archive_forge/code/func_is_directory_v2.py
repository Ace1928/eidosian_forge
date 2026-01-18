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
@tf_export('io.gfile.isdir')
def is_directory_v2(path):
    """Returns whether the path is a directory or not.

  Args:
    path: string, path to a potential directory

  Returns:
    True, if the path is a directory; False otherwise
  """
    try:
        return _pywrap_file_io.IsDirectory(compat.path_to_bytes(path))
    except errors.OpError:
        return False